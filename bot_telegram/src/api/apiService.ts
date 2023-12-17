import axios from "axios";
import {apiLink} from '../config/configFile';

export const postToApi = async (text: string) => {
    if (apiLink) {
        return await axios.post(apiLink, {text});
    }
};

export const getFromApi = async (requestId: number) => {
    return await axios.get(`${apiLink}/${requestId}`);
};
