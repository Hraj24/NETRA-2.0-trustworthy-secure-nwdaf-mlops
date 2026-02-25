import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// fetch(`${API_BASE_URL}/predict`);

export const predict = (data) =>
  axios.post(`${API_BASE_URL}/predict`, data);

export const explain = (data) =>
  axios.post(`${API_BASE_URL}/explain`, data);

export const health = () =>
  axios.get(`${API_BASE_URL}/health`);
