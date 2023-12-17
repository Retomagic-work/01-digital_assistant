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

export const getRequestFilter = async (
  date_from: string,
  date_to: string,
  headers: HeadersInit = {}
) => {
  try {
    const token = getToken();

    const params = new URLSearchParams({
      date_from: date_from,
      date_to: date_to,
      //   rating: rating,
    }).toString();

    const response = await fetch(`${endpoint}/requests/dump${params}`, {
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

// export const editRequest = async ({ login, password }: UserData) => {
//   try {
//     const response = await fetch(`${endpoint}/auth/login`, {
//       method: "POST",
//       headers: {
//         "Content-Type": "application/json",
//       },
//       body: JSON.stringify({ login, password }),
//     });
//     if (!response.ok) {
//       throw new Error("Network response wasn`t ok");
//     }
//     const data = await response.json();
//     return data;
//   } catch (error) {
//     console.log("There was a problem with the fetch operation:", error);
//     throw error;
//   }
// };

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
