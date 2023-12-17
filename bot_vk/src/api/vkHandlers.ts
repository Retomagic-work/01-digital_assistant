import {getFromApi, postToApi} from './apiService';

interface IMessageContext {
    text?: string;
    hasText: boolean;
    send: (message: string) => Promise<any>;
}


interface IApiResponse {
    data: {
        id: number;
        status: string;
        response: {
            text: string;
        };
    };
}


export const handleMessage = async (context: IMessageContext) => {
    if (!context.hasText || !context.text) return;

    try {
        const postResponse = await postToApi(context.text) as IApiResponse; // Cast the response to IApiResponse
        if (postResponse && postResponse.data.id) {
            const requestId = postResponse.data.id;
            await checkResponse(context, requestId);
        }
    } catch (error) {
        console.error(error);
        await context.send(`Произошла ошибка: ${error}`);
    }
};

const checkResponse = async (context: IMessageContext, requestId: number) => {
    const interval = setInterval(async () => {
        try {
            const getResponse = await getFromApi(requestId) as IApiResponse; // Cast the response to IApiResponse
            if (getResponse.data.status === 'success') {
                clearInterval(interval);
                await context.send(getResponse.data.response.text);
            }
        } catch (error) {
            clearInterval(interval);
            console.error(error);
            await context.send(`Произошла ошибка при получении ответа: ${error}`);
        }
    }, 500);
};
