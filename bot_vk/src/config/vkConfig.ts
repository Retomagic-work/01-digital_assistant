import { VK } from 'vk-io';
import 'dotenv/config';

export const vk = new VK({
    token: process.env.BOT_TOKEN as string
});

if (!process.env.BOT_TOKEN) {
    console.error('BOT_TOKEN is not defined in the environment.');
    process.exit(1);
}