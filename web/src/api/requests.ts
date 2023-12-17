import { saveAs } from "file-saver";

import { formatYear } from "../utils/formatYear";
import { getToken } from "../utils/getToken";

const endpoint = process.env.REACT_APP_API_ENDPOINT;

export const getRequest = async (headers: HeadersInit = {}) => {
  try {
    const token = getToken();

    const response = await fetch(`${endpoint}/requests`, {
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

export const requestSave = async (
  dateFrom: string,
  dateTo: string,
  rating: number | null,
  headers: HeadersInit = {}
) => {
  try {
    const token = getToken();

    const date_from = formatYear(dateFrom);
    const date_to = formatYear(dateTo);

    const params = new URLSearchParams({
      date_from: date_from.toString(),
      date_to: date_to.toString(),
    }).toString();

    const response = await fetch(
      `${endpoint}/requests/download/dump?${params}&${
        rating ? `rating=${rating}` : ""
      }`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
          ...(token && {
            Authorization: `Bearer ${token}`,
          }),
          ...headers,
        },
      }
    );
    if (!response.ok) {
      throw new Error("Network response wasn`t ok");
    }

    const data = await response.json();

    saveAs(
      new Blob([JSON.stringify(data, null, 2)], {
        type: "application/json;charset=" + document.characterSet,
      }),
      `archive-${date_from} - ${date_to}.json`
    );
  } catch (error) {
    console.log("There was a problem with the fetch operation:", error);
    throw error;
  }
};

export const putRating = async (
  id: number,
  rating: number,
  headers: HeadersInit = {}
) => {
  try {
    const token = getToken();

    const response = await fetch(`${endpoint}/requests/${id}`, {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...(token && {
          Authorization: `Bearer ${token}`,
        }),
        ...headers,
      },
      body: JSON.stringify({ rating }),
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
