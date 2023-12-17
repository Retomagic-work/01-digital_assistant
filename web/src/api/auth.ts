import { UserData } from "../types/Auth";

const endpoint = process.env.REACT_APP_API_ENDPOINT;

export const authorizationUser = async ({ login, password }: UserData) => {
  try {
    const response = await fetch(`${endpoint}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ login, password }),
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
