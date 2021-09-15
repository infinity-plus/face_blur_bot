from telegram import Message, Update
from telegram.ext import CallbackContext, CommandHandler, Filters, Updater, MessageHandler

from .face_blur import face_blur
from . import APP_NAME, BOT_TOKEN, PORT
import os


def start(update: Update, _: CallbackContext):
    if isinstance(update.effective_message, Message):
        update.effective_message.reply_text(
            'Hi! I am a bot that can blur faces in photos.\n' +
            'Just send me a photo and I will send you back a blurred one!')


def blur(update: Update, _: CallbackContext):
    if isinstance(update.effective_message, Message):
        document = update.effective_message.document.get_file().download()
        new_doc = face_blur(document)
        update.effective_message.reply_document(open(new_doc, 'rb'))
        if os.path.isfile(new_doc):
            os.remove(new_doc)


updater = Updater(token=BOT_TOKEN)
updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(
    MessageHandler(Filters.document.category("image"), blur))

updater.start_webhook(listen="0.0.0.0",
                      port=PORT,
                      url_path=BOT_TOKEN,
                      webhook_url="https://{}.herokuapp.com/{}".format(
                          APP_NAME, BOT_TOKEN))
