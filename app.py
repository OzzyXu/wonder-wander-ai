import asyncio
import copy
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone

import httpx
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azure.identity import DefaultAzureCredential
from azure.cosmos.aio import CosmosClient  # Async client
from azure.identity.aio import get_bearer_token_provider
from azure.storage.blob.aio import BlobServiceClient
from openai import AsyncAzureOpenAI
from quart import (
    Blueprint,
    Quart,
    current_app,
    jsonify,
    make_response,
    render_template,
    request,
    send_from_directory,
)

from backend.auth.auth_utils import get_authenticated_user_details
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.security.ms_defender_utils import get_msdefender_user_json
from backend.settings import (
    MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION,
    app_settings,
)
from backend.utils import (
    convert_to_pf_format,
    format_as_ndjson,
    format_non_streaming_response,
    format_pf_non_streaming_response,
    format_stream_response,
)

logger = logging.getLogger(__name__)

bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

cosmos_db_ready = asyncio.Event()

# Load from App Settings (only account and container names needed)
blob_account_name = app_settings.blobstorage.account_name
blob_container_name = app_settings.blobstorage.container_name

# Initialize Blob Service Client
blob_account_url = f"https://{blob_account_name}.blob.core.windows.net"
blob_credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(
    account_url=blob_account_url, credential=blob_credential
)
blob_container_client = blob_service_client.get_container_client(blob_container_name)

# Cosmos DB and Blob Storage configuration
photo_cosmos_url = app_settings.chat_history.account_url
photo_cosmos_key = app_settings.chat_history.account_key

# Validate Cosmos DB credentials
if not photo_cosmos_url:
    raise ValueError("COSMOS_URL not set in environment variables")
if not photo_cosmos_key:
    raise ValueError("COSMOS_KEY not set in environment variables")
if not photo_cosmos_url.startswith("https://"):
    raise ValueError("COSMOS_URL must start with 'https://'")


def create_app():
    app = Quart(__name__)
    app.register_blueprint(bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    @app.before_serving
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()

            ### TODO: this is not good and needs to get updated later.
            global photo_cosmos_client, photo_cosmos_database, photo_cosmos_container
            photo_cosmos_client = CosmosClient(
                photo_cosmos_url, credential=photo_cosmos_key
            )
            photo_cosmos_database = photo_cosmos_client.get_database_client("db_photo")
            photo_cosmos_container = photo_cosmos_database.get_container_client(
                "metadata"
            )
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise e

    return app


@bp.route("/")
async def index():
    return await render_template(
        "index.html", title=app_settings.ui.title, favicon=app_settings.ui.favicon
    )


@bp.route("/favicon.ico")
async def favicon():
    return await bp.send_static_file("favicon.ico")


@bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)


# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"


# Frontend Settings via Environment Variables
frontend_settings = {
    "auth_enabled": app_settings.base_settings.auth_enabled,
    "feedback_enabled": (
        app_settings.chat_history and app_settings.chat_history.enable_feedback
    ),
    "ui": {
        "title": app_settings.ui.title,
        "logo": app_settings.ui.logo,
        "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
        "chat_title": app_settings.ui.chat_title,
        "chat_description": app_settings.ui.chat_description,
        "show_share_button": app_settings.ui.show_share_button,
        "show_chat_history_button": app_settings.ui.show_chat_history_button,
    },
    "sanitize_answer": app_settings.base_settings.sanitize_answer,
    "oyd_enabled": app_settings.base_settings.datasource_type,
}


# Enable Microsoft Defender for Cloud Integration
MS_DEFENDER_ENABLED = os.environ.get("MS_DEFENDER_ENABLED", "true").lower() == "true"


# Initialize Azure OpenAI Client
async def init_openai_client():
    azure_openai_client = None

    try:
        # API version check
        if (
            app_settings.azure_openai.preview_api_version
            < MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
        ):
            raise ValueError(
                f"The minimum supported Azure OpenAI preview API version is '{MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION}'"
            )

        # Endpoint
        if (
            not app_settings.azure_openai.endpoint
            and not app_settings.azure_openai.resource
        ):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required"
            )

        endpoint = (
            app_settings.azure_openai.endpoint
            if app_settings.azure_openai.endpoint
            else f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        )

        # Authentication
        aoai_api_key = app_settings.azure_openai.key
        ad_token_provider = None
        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure Entra ID auth")
            async with DefaultAzureCredential() as credential:
                ad_token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )

        # Deployment
        deployment = app_settings.azure_openai.model
        if not deployment:
            raise ValueError("AZURE_OPENAI_MODEL is required")

        # Default Headers
        default_headers = {"x-ms-useragent": USER_AGENT}

        azure_openai_client = AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_ad_token_provider=ad_token_provider,
            default_headers=default_headers,
            azure_endpoint=endpoint,
        )

        return azure_openai_client
    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization", e)
        azure_openai_client = None
        raise e


async def init_cosmosdb_client():
    cosmos_conversation_client = None
    if app_settings.chat_history:
        try:
            cosmos_endpoint = (
                f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
            )

            if not app_settings.chat_history.account_key:
                async with DefaultAzureCredential() as cred:
                    credential = cred

            else:
                credential = app_settings.chat_history.account_key

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            cosmos_conversation_client = None
            raise e
    else:
        logging.debug("CosmosDB not configured")

    return cosmos_conversation_client


def prepare_model_args(request_body, request_headers):
    request_messages = request_body.get("messages", [])
    messages = []
    if not app_settings.datasource:
        messages = [
            {"role": "system", "content": app_settings.azure_openai.system_message}
        ]

    for message in request_messages:
        if message:
            if message["role"] == "assistant" and "context" in message:
                context_obj = json.loads(message["context"])
                messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "context": context_obj,
                    }
                )
            else:
                messages.append(
                    {"role": message["role"], "content": message["content"]}
                )

    user_json = None
    if MS_DEFENDER_ENABLED:
        authenticated_user_details = get_authenticated_user_details(request_headers)
        conversation_id = request_body.get("conversation_id", None)
        application_name = app_settings.ui.title
        user_json = get_msdefender_user_json(
            authenticated_user_details,
            request_headers,
            conversation_id,
            application_name,
        )

    model_args = {
        "messages": messages,
        "temperature": app_settings.azure_openai.temperature,
        "max_tokens": app_settings.azure_openai.max_tokens,
        "top_p": app_settings.azure_openai.top_p,
        "stop": app_settings.azure_openai.stop_sequence,
        "stream": app_settings.azure_openai.stream,
        "model": app_settings.azure_openai.model,
        "user": user_json,
    }

    if app_settings.datasource:
        model_args["extra_body"] = {
            "data_sources": [
                app_settings.datasource.construct_payload_configuration(request=request)
            ]
        }

    model_args_clean = copy.deepcopy(model_args)
    if model_args_clean.get("extra_body"):
        secret_params = [
            "key",
            "connection_string",
            "embedding_key",
            "encoded_api_key",
            "api_key",
        ]
        for secret_param in secret_params:
            if model_args_clean["extra_body"]["data_sources"][0]["parameters"].get(
                secret_param
            ):
                model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                    secret_param
                ] = "*****"
        authentication = model_args_clean["extra_body"]["data_sources"][0][
            "parameters"
        ].get("authentication", {})
        for field in authentication:
            if field in secret_params:
                model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                    "authentication"
                ][field] = "*****"
        embeddingDependency = model_args_clean["extra_body"]["data_sources"][0][
            "parameters"
        ].get("embedding_dependency", {})
        if "authentication" in embeddingDependency:
            for field in embeddingDependency["authentication"]:
                if field in secret_params:
                    model_args_clean["extra_body"]["data_sources"][0]["parameters"][
                        "embedding_dependency"
                    ]["authentication"][field] = "*****"

    logging.debug(f"REQUEST BODY: {json.dumps(model_args_clean, indent=4)}")

    return model_args


async def promptflow_request(request):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {app_settings.promptflow.api_key}",
        }
        # Adding timeout for scenarios where response takes longer to come back
        logging.debug(f"Setting timeout to {app_settings.promptflow.response_timeout}")
        async with httpx.AsyncClient(
            timeout=float(app_settings.promptflow.response_timeout)
        ) as client:
            pf_formatted_obj = convert_to_pf_format(
                request,
                app_settings.promptflow.request_field_name,
                app_settings.promptflow.response_field_name,
            )
            # NOTE: This only support question and chat_history parameters
            # If you need to add more parameters, you need to modify the request body
            response = await client.post(
                app_settings.promptflow.endpoint,
                json={
                    app_settings.promptflow.request_field_name: pf_formatted_obj[-1][
                        "inputs"
                    ][app_settings.promptflow.request_field_name],
                    "chat_history": pf_formatted_obj[:-1],
                },
                headers=headers,
            )
        resp = response.json()
        resp["id"] = request["messages"][-1]["id"]
        return resp
    except Exception as e:
        logging.error(f"An error occurred while making promptflow_request: {e}")


async def send_chat_request(request_body, request_headers):
    filtered_messages = []
    messages = request_body.get("messages", [])
    for message in messages:
        if message.get("role") != "tool":
            filtered_messages.append(message)

    request_body["messages"] = filtered_messages
    model_args = prepare_model_args(request_body, request_headers)

    try:
        azure_openai_client = await init_openai_client()
        raw_response = (
            await azure_openai_client.chat.completions.with_raw_response.create(
                **model_args
            )
        )
        response = raw_response.parse()
        apim_request_id = raw_response.headers.get("apim-request-id")
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e

    return response, apim_request_id


async def complete_chat_request(request_body, request_headers):
    if app_settings.base_settings.use_promptflow:
        response = await promptflow_request(request_body)
        history_metadata = request_body.get("history_metadata", {})
        return format_pf_non_streaming_response(
            response,
            history_metadata,
            app_settings.promptflow.response_field_name,
            app_settings.promptflow.citations_field_name,
        )
    else:
        response, apim_request_id = await send_chat_request(
            request_body, request_headers
        )
        history_metadata = request_body.get("history_metadata", {})
        return format_non_streaming_response(
            response, history_metadata, apim_request_id
        )


async def stream_chat_request(request_body, request_headers):
    response, apim_request_id = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {})

    async def generate():
        async for completionChunk in response:
            yield format_stream_response(
                completionChunk, history_metadata, apim_request_id
            )

    return generate()


async def conversation_internal(request_body, request_headers):
    try:
        if (
            app_settings.azure_openai.stream
            and not app_settings.base_settings.use_promptflow
        ):
            result = await stream_chat_request(request_body, request_headers)
            response = await make_response(format_as_ndjson(result))
            response.timeout = None
            response.mimetype = "application/json-lines"
            return response
        else:
            result = await complete_chat_request(request_body, request_headers)
            return jsonify(result)

    except Exception as ex:
        logging.exception(ex)
        if hasattr(ex, "status_code"):
            return jsonify({"error": str(ex)}), ex.status_code
        else:
            return jsonify({"error": str(ex)}), 500


@bp.route("/conversation", methods=["POST"])
async def conversation():
    logging.info("conversation_endpoint was called")
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()

    return await conversation_internal(request_json, request.headers)


# ----------------------------------------------------------------------
# Cleaning function: removes extra spaces before punctuation and
# ensures one space after punctuation.
# ----------------------------------------------------------------------
def clean_aggregated_content(text: str) -> str:
    # Remove leading/trailing whitespace.
    text = text.strip()
    # Remove spaces before punctuation symbols (comma, period, exclamation, question mark).
    text = re.sub(r"\s+([,!.?])", r"\1", text)
    # Ensure exactly one space after punctuation.
    text = re.sub(r"([,!.?])\s*", r"\1 ", text)
    # Remove any trailing space added at the end.
    return text.strip()


# ----------------------------------------------------------------------
# Aggregation helper function.
#
# This function aggregates all NDJSON chunks into a single JSON object.
# It concatenates all the "content" fields from the "choices" messages,
# cleans the aggregated string, and then uses the first valid JSON chunk
# as a base to preserve all other fields.
# ----------------------------------------------------------------------
async def aggregate_conversation_data(response) -> dict:
    aggregated_message = ""
    base_data = None

    # Check for a streaming response.
    if response.mimetype == "application/json-lines" and hasattr(
        response.response, "__aiter__"
    ):
        async for chunk in response.response:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            for line in chunk.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("object") == "chat.completion.chunk":
                    if base_data is None:
                        base_data = data.copy()
                    for choice in data.get("choices", []):
                        for message in choice.get("messages", []):
                            aggregated_message += message.get("content", "")
    else:
        # Non-streaming: Read the full response body.
        data_str = await response.get_data(as_text=True)
        lines = data_str.splitlines()
        if len(lines) > 1:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("object") == "chat.completion.chunk":
                    if base_data is None:
                        base_data = data.copy()
                    for choice in data.get("choices", []):
                        for message in choice.get("messages", []):
                            aggregated_message += message.get("content", "")
        else:
            try:
                data = json.loads(data_str)
                if data.get("object") == "chat.completion.chunk" and "choices" in data:
                    base_data = data.copy()
                    for choice in data.get("choices", []):
                        for message in choice.get("messages", []):
                            aggregated_message += message.get("content", "")
                else:
                    return data
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response"}

    if base_data is not None:
        # Clean the aggregated message to remove unwanted spaces before punctuation.
        aggregated_message = clean_aggregated_content(aggregated_message)
        for choice in base_data.get("choices", []):
            for message in choice.get("messages", []):
                message["content"] = aggregated_message
        return base_data
    else:
        return {"aggregated_message": clean_aggregated_content(aggregated_message)}


# ----------------------------------------------------------------------
# New endpoint: Calls conversation_internal and returns the aggregated JSON.
# ----------------------------------------------------------------------
@bp.route("/aggregate_conversation", methods=["POST"])
async def aggregate_conversation_endpoint():
    logging.info("aggregate_conversation_endpoint was called")
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415

    request_json = await request.get_json()

    # Call conversation_internal to get a Response object.
    response = await conversation_internal(request_json, request.headers)

    # Aggregate the response content.
    aggregated_data = await aggregate_conversation_data(response)

    # Use custom separators to ensure a space after each comma and colon.
    formatted_json = json.dumps(aggregated_data, separators=(", ", ": "))
    formatted_json += "\n"
    return formatted_json, 200, {"Content-Type": "application/json"}


@bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500


## Conversation History API ##
@bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"])
            conversation_dict = (
                await current_app.cosmos_conversation_client.create_conversation(
                    user_id=user_id, title=title
                )
            )
            conversation_id = conversation_dict["id"]
            history_metadata["title"] = title
            history_metadata["date"] = conversation_dict["createdAt"]

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "user":
            createdMessageValue = (
                await current_app.cosmos_conversation_client.create_message(
                    uuid=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-1],
                )
            )
            if createdMessageValue == "Conversation not found":
                raise Exception(
                    "Conversation not found for the given conversation ID: "
                    + conversation_id
                    + "."
                )
        else:
            raise Exception("No user message found")

        # Submit request to Chat Completions for response
        request_body = await request.get_json()
        history_metadata["conversation_id"] = conversation_id
        request_body["history_metadata"] = history_metadata
        return await conversation_internal(request_body, request.headers)

    except Exception as e:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/update", methods=["POST"])
async def update_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "assistant":
            if len(messages) > 1 and messages[-2].get("role", None) == "tool":
                # write the tool message first
                await current_app.cosmos_conversation_client.create_message(
                    uuid=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-2],
                )
            # write the assistant message
            await current_app.cosmos_conversation_client.create_message(
                uuid=messages[-1]["id"],
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
        else:
            raise Exception("No bot messages found")

        # Submit request to Chat Completions for response
        response = {"success": True}
        return jsonify(response), 200

    except Exception as e:
        logging.exception("Exception in /history/update")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/message_feedback", methods=["POST"])
async def update_message():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for message_id
    request_json = await request.get_json()
    message_id = request_json.get("message_id", None)
    message_feedback = request_json.get("message_feedback", None)
    try:
        if not message_id:
            return jsonify({"error": "message_id is required"}), 400

        if not message_feedback:
            return jsonify({"error": "message_feedback is required"}), 400

        ## update the message in cosmos
        updated_message = (
            await current_app.cosmos_conversation_client.update_message_feedback(
                user_id, message_id, message_feedback
            )
        )
        if updated_message:
            return (
                jsonify(
                    {
                        "message": f"Successfully updated message with feedback {message_feedback}",
                        "message_id": message_id,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "error": f"Unable to update message {message_id}. It either does not exist or the user does not have access to it."
                    }
                ),
                404,
            )

    except Exception as e:
        logging.exception("Exception in /history/message_feedback")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/delete", methods=["DELETE"])
async def delete_conversation():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos first
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        ## Now delete the conversation
        deleted_conversation = (
            await current_app.cosmos_conversation_client.delete_conversation(
                user_id, conversation_id
            )
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted conversation and messages",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/delete")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/list", methods=["GET"])
async def list_conversations():
    await cosmos_db_ready.wait()
    offset = request.args.get("offset", 0)
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversations from cosmos
    conversations = await current_app.cosmos_conversation_client.get_conversations(
        user_id, offset=offset, limit=25
    )
    if not isinstance(conversations, list):
        return jsonify({"error": f"No conversations for {user_id} were found"}), 404

    ## return the conversation ids

    return jsonify(conversations), 200


@bp.route("/history/read", methods=["POST"])
async def get_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation object and the related messages from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    ## return the conversation id and the messages in the bot frontend format
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    # get the messages for the conversation from cosmos
    conversation_messages = await current_app.cosmos_conversation_client.get_messages(
        user_id, conversation_id
    )

    ## format the messages in the bot frontend format
    messages = [
        {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "createdAt": msg["createdAt"],
            "feedback": msg.get("feedback"),
        }
        for msg in conversation_messages
    ]

    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200


@bp.route("/history/rename", methods=["POST"])
async def rename_conversation():
    await cosmos_db_ready.wait()
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    ## update the title
    title = request_json.get("title", None)
    if not title:
        return jsonify({"error": "title is required"}), 400
    conversation["title"] = title
    updated_conversation = (
        await current_app.cosmos_conversation_client.upsert_conversation(conversation)
    )

    return jsonify(updated_conversation), 200


@bp.route("/history/delete_all", methods=["DELETE"])
async def delete_all_conversations():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    # get conversations for user
    try:
        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        conversations = await current_app.cosmos_conversation_client.get_conversations(
            user_id, offset=0, limit=None
        )
        if not conversations:
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404

        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from cosmos first
            deleted_messages = (
                await current_app.cosmos_conversation_client.delete_messages(
                    conversation["id"], user_id
                )
            )

            ## Now delete the conversation
            deleted_conversation = (
                await current_app.cosmos_conversation_client.delete_conversation(
                    user_id, conversation["id"]
                )
            )
        return (
            jsonify(
                {
                    "message": f"Successfully deleted conversation and messages for user {user_id}"
                }
            ),
            200,
        )

    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/clear", methods=["POST"])
async def clear_messages():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    user_id = authenticated_user["user_principal_id"]

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted messages in conversation",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        return jsonify({"error": str(e)}), 500


@bp.route("/history/ensure", methods=["GET"])
async def ensure_cosmos():
    await cosmos_db_ready.wait()
    if not app_settings.chat_history:
        return jsonify({"error": "CosmosDB is not configured"}), 404

    try:
        success, err = await current_app.cosmos_conversation_client.ensure()
        if not current_app.cosmos_conversation_client or not success:
            if err:
                return jsonify({"error": err}), 422
            return jsonify({"error": "CosmosDB is not configured or not working"}), 500

        return jsonify({"message": "CosmosDB is configured and working"}), 200
    except Exception as e:
        logging.exception("Exception in /history/ensure")
        cosmos_exception = str(e)
        if "Invalid credentials" in cosmos_exception:
            return jsonify({"error": cosmos_exception}), 401
        elif "Invalid CosmosDB database name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception} {app_settings.chat_history.database} for account {app_settings.chat_history.account}"
                    }
                ),
                422,
            )
        elif "Invalid CosmosDB container name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception}: {app_settings.chat_history.conversations_container}"
                    }
                ),
                422,
            )
        else:
            return jsonify({"error": "CosmosDB is not working"}), 500


async def generate_title(conversation_messages) -> str:
    ## make sure the messages are sorted by _ts descending
    title_prompt = "Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Do not include any other commentary or description."

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_messages
    ]
    messages.append({"role": "user", "content": title_prompt})

    try:
        azure_openai_client = await init_openai_client()
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model,
            messages=messages,
            temperature=1,
            max_tokens=64,
        )

        title = response.choices[0].message.content
        return title
    except Exception as e:
        logging.exception("Exception while generating title", e)
        return messages[-2]["content"]


# Function to get blob URLs as a dictionary (async)
async def get_blob_urls(container_name="photos", sas_duration_hours=1):
    """
    Returns a dictionary of blob names mapped to their URLs.
    Args:
        container_name (str): Name of the container (default: 'photos').
        sas_duration_hours (int): Duration in hours for SAS token validity (default: 1).
    Returns:
        dict: Dictionary with blob names as keys and URLs as values.
    """
    # Get user delegation key asynchronously
    start_time = datetime.now(timezone.utc) - timedelta(minutes=15)
    expiry_time = datetime.now(timezone.utc) + timedelta(hours=1)
    user_delegation_key = await blob_service_client.get_user_delegation_key(
        key_start_time=start_time,
        key_expiry_time=expiry_time
    )
    
    # Get container client synchronously
    container_client = blob_service_client.get_container_client(container_name)
    
    # List blobs asynchronously
    blob_list = []
    async for blob in container_client.list_blobs():
        blob_list.append(blob)
    
    url_dict = {}
    for blob in blob_list:
        blob_name = blob.name
        blob_client = container_client.get_blob_client(blob_name)

        # Generate SAS URL with user delegation key
        sas_token = generate_blob_sas(
            account_name=blob_account_name,
            container_name=container_name,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True),  # Read-only
            expiry=datetime.now(timezone.utc) + timedelta(hours=sas_duration_hours),
        )
        sas_url = f"{blob_client.url}?{sas_token}"
        url_dict[blob_name] = sas_url

    return url_dict

# Updated route to list all photo URLs in "name": "url" format (async)
@bp.route("/photos", methods=["GET"])
async def list_photos():
    try:
        url_dict = await get_blob_urls(blob_container_name)
        logger.info("Listed %d photo URLs", len(url_dict))
        return jsonify({"photos": url_dict})
    except Exception as e:
        logger.error("Error listing photos: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.route("/api/sas", methods=["GET"])
async def get_sas():
    try:
        blob_name = request.args.get("blob_name")
        # Get user delegation key asynchronously
        start_time = datetime.now(timezone.utc) - timedelta(
            minutes=15
        )  # Allow for clock skew
        expiry_time = datetime.now(timezone.utc) + timedelta(hours=1)
        user_delegation_key = await blob_service_client.get_user_delegation_key(
            key_start_time=start_time, key_expiry_time=expiry_time
        )

        logger.debug(
            "User Delegation Key generated: value=%s", user_delegation_key.value
        )

        blob_client = blob_service_client.get_blob_client(
            blob_container_name, blob_name
        )

        # Generate User Delegation SAS token
        sas_token = generate_blob_sas(
            account_name=blob_account_name,
            container_name=blob_container_name,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(write=True),
            expiry=datetime.now(timezone.utc) + timedelta(minutes=15),
        )

        sas_url = f"{blob_client.url}?{sas_token}"
        logger.info("Generated SAS URL: %s", sas_url)
        logger.info("SAS Token: %s", sas_token)
        return jsonify({"sas_url": sas_url})
    except Exception as e:
        logger.error("Error in /api/sas: %s", e)
        return jsonify({"error": str(e)}), 500


@bp.route("/upload", methods=["POST"])
async def track_photo():
    try:
        data = await request.get_json()  # Async JSON parsing
        photo_entry = {
            "id": data["blob_name"],
            "user_id": data["user_id"],
            "conversation_id": data.get("conversation_id", "test_conv"),
            "url": data["url"],
            "timestamp": data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        }
        await photo_cosmos_container.upsert_item(photo_entry)  # Async upsert
        return jsonify({"message": "Photo tracked"})
    except Exception as e:
        print(f"Error in /upload: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/photos/<user_id>", methods=["GET"])
async def get_user_photos(user_id):
    try:
        query = "SELECT c.id, c.url, c.conversation_id, c.timestamp FROM c WHERE c.user_id = @user_id"
        photos = []
        async for item in photo_cosmos_container.query_items(
            query=query, parameters=[{"name": "@user_id", "value": user_id}]
        ):
            photos.append(item)
        return jsonify({"photos": photos})
    except Exception as e:
        print(f"Error in /photos: {e}")
        return jsonify({"error": str(e)}), 500


@bp.route("/test-photo-cosmos", methods=["GET"])
def test_photo_cosmos():
    try:
        photo_cosmos_container.read()  # Checks if container is accessible
        return jsonify({"message": "Cosmos DB connection successful"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


app = create_app()
