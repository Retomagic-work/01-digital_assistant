import { vk } from './src/config/vkConfig';
import { handleMessage } from './src/api/vkHandlers';

vk.updates.on('message_new', async (context) => {
    if (context.isOutbox) return;
    await handleMessage(context);
});

vk.updates.start().catch(console.error);