import os
import logging
import requests
import json
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

with open("service_details.json", "r") as f:
    service_details = json.load(f)

SCORING_URI = service_details["scoring_uri"]
API_KEY = service_details["primary_key"]

TOKEN = "7864119630:AAHWhaYiODni7j_GmV0EacmSTIe-bN9JcZ0"

def start(update: Update, context: CallbackContext) -> None:
    user = update.effective_user
    update.message.reply_text(
        f'¡Hola {user.first_name}! Soy el bot de MESCYT. Puedes hacerme preguntas sobre '
        'legalización de documentos, becas, y el programa TIEs.'
    )

def help_command(update: Update, context: CallbackContext) -> None:
    update.message.reply_text(
        'Puedes preguntarme sobre:\n'
        '- Legalización de documentos académicos\n'
        '- Programa de Incentivo Estudiantil (TIEs)\n'
        '- Becas nacionales e internacionales\n\n'
        'Solo escribe tu pregunta y te responderé lo mejor posible.'
    )

def query_ml_endpoint(question: str) -> dict:
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    data = {
        "question": question
    }

    try:
        logger.info(f"Sending request to {SCORING_URI}")
        response = requests.post(SCORING_URI, data=json.dumps(data), headers=headers)
        
        if response.status_code == 200:
            logger.info(f"Received response: {response.text}")
            
            try:
                result_json = json.loads(response.text)
                
                if isinstance(result_json, str):
                    try:
                        result_json = json.loads(result_json)
                        logger.info(f"Parsed nested JSON: {result_json}")
                    except json.JSONDecodeError:
                        logger.warning("Could not parse nested JSON, using as is")
                
                return result_json
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {response.text}")
                return {"answer": "Lo siento, hubo un error al procesar la respuesta."}
        else:
            logger.error(f"Error from ML endpoint: {response.status_code} - {response.text}")
            return {"answer": "Lo siento, estoy teniendo problemas técnicos en este momento."}
    except Exception as e:
        logger.error(f"Exception when calling ML endpoint: {str(e)}")
        return {"answer": "Lo siento, estoy teniendo problemas técnicos en este momento."}

def handle_message(update: Update, context: CallbackContext) -> None:
    question = update.message.text
    logger.info(f"User sent message: {question}")

    update.message.reply_text("Procesando tu pregunta...")

    result = query_ml_endpoint(question)
    logger.info(f"Result type: {type(result)}, content: {result}")

    if isinstance(result, dict):
        answer = result.get("answer", "Lo siento, no pude encontrar una respuesta.")
        confidence = result.get("confidence", 0)
        
        if confidence > 0.8:
            update.message.reply_text(answer)
        elif confidence > 0.5:
            update.message.reply_text(f"{answer}\n\n(Nota: Estoy moderadamente seguro de esta respuesta)")
        else:
            update.message.reply_text(f"No estoy muy seguro, pero esto podría ayudar:\n\n{answer}")
        
        if "matched_question" in result:
            matched_question = result["matched_question"]
            update.message.reply_text(f"Tu pregunta se parece a: '{matched_question}', la cual es una pregunta frecuente.")
    else:
        update.message.reply_text(f"Respuesta: {str(result)}")

def main() -> None:
    try:
        updater = Updater(TOKEN)
        dispatcher = updater.dispatcher
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))
        dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
        logger.info("Starting bot...")
        updater.start_polling()
        logger.info("Bot started. Press Ctrl+C to stop.")
        updater.idle()
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        raise

if __name__ == '__main__':
    main()
