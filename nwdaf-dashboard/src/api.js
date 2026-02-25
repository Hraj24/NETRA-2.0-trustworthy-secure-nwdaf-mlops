import axios from "axios";

// const API_BASE = "http://backend:8000"; // Docker network name
// const API = "http://127.0.0.1:8000";
const API_BASE_URL = import.meta.env.VITE_API_URL;

// fetch(`${API_BASE_URL}/predict`);

export const predict = (data) =>
  axios.post(`${API_BASE_URL}/predict`, data);

export const explain = (data) =>
  axios.post(`${API_BASE_URL}/explain`, data);

export const health = () =>
  axios.get(`${API_BASE_URL}/health`);
