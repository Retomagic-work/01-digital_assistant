import {postToApi, getFromApi} from "../api/apiService";
import axios from "axios";

export const handleTextMessage = async (ctx: any) => {
    try {
        const postResponse = await postToApi(ctx.message.text);

        if (postResponse) {
            const requestId = postResponse.data.id

            const checkResponse = async () => {
                try {
                    const getResponse = await axios.get(`${process.env.API_LINK}/${requestId}`)
                    if (getResponse.data.status === 'success') {
                        clearInterval(checkInterval)
                        const replyMessage = getResponse.data.response.text;
                        ctx.reply(replyMessage)
                        ctx.telegram.sendChatAction(ctx.chat.id, 'typing')
                    } else {
                        ctx.telegram.sendChatAction(ctx.chat.id, 'typing')
                    }
                } catch (error) {
                    clearInterval(checkInterval)
                    console.error(error)
                    ctx.reply(`Произошла ошибка при получении ответа: ${error}`)
                }
            }
            const checkInterval = setInterval(checkResponse, 500)
        }

    } catch (error) {
        console.error(error)
        ctx.reply(`Произошла ошибка: ${error}`)
    }
}

export const handleStickerMessage = async (ctx: any) => {
    const stickerFileId = ctx.message.sticker.file_id
    ctx.replyWithSticker(stickerFileId)
}
