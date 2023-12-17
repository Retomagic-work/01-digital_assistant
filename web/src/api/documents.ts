import { getToken } from "../utils/getToken";

const endpoint = process.env.REACT_APP_API_ENDPOINT;

export const getDocuments = async (headers: HeadersInit = {}) => {
  try {
    const token = getToken();

    const response = await fetch(`${endpoint}/documents`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        ...(token && {
          Authorization: `Bearer ${token}`,
        }),
        ...headers,
      },
    });
    if (!response.ok) {
      throw new Error("Network response wasn`t ok");
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.log("There was a problem with the fetch operation:", error);
    throw error;
  }
};

export const createDocuments = async (
  files: any,
  headers: HeadersInit = {}
) => {
  try {
    const token = getToken();

    const formData = new FormData();
    files.forEach((file: any) => formData.append("files", file));

    const response = await fetch(`${endpoint}/documents`, {
      method: "POST",
      headers: {
        ...(token && {
          Authorization: `Bearer ${token}`,
        }),
        ...headers,
      },
      body: formData,
    });
    if (!response.ok) {
      throw new Error("Network response wasn`t ok");
    }
    const data = await response.json();

    return data;
  } catch (error) {
    console.log("There was a problem with the fetch operation:", error);
    throw error;
  }
};

export const putDescriptionDocument = async (
  id: number,
  description: string,
  headers: HeadersInit = {}
) => {
  try {
    const token = getToken();

    const response = await fetch(`${endpoint}/documents/${id}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...(token && {
          Authorization: `Bearer ${token}`,
        }),
        ...headers,
      },
      body: JSON.stringify({ description }),
    });
    if (!response.ok) {
      throw new Error("Network response wasn`t ok");
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.log("There was a problem with the fetch operation:", error);
    throw error;
  }
};
