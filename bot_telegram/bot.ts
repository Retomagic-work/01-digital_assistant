import {Telegraf} from 'telegraf';
import 'dotenv/config';
import {handleStickerMessage, handleTextMessage} from "./src/messageHandlers/messageHandlers";

const botToken = process.env.BOT_TOKEN;
if (!botToken) {
    throw new Error('BOT_TOKEN must be provided!');
}

const bot = new Telegraf(botToken);

bot.start(ctx => {
    ctx.reply('Привет! Я Умный цифровой помощник Главы администрации города Мирный!');
});

bot.on('text', handleTextMessage);

bot.on('sticker', handleStickerMessage)


bot.launch();

process.once('SIGINT', () => bot.stop('SIGINT'));
process.once('SIGTERM', () => bot.stop('SIGTERM'));
