// NETRA 1.0 | IMT-2030 Compliant 6G NWDAF Dashboard
// Citation: ITU-R M.2160 / IMT-2030 Framework

import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const predict = (data) =>
  axios.post(`${API_BASE_URL}/predict`, data);

export const explain = (data) =>
  axios.post(`${API_BASE_URL}/explain`, data);

export const health = () =>
  axios.get(`${API_BASE_URL}/health`);

export const driftLog = async () => {
  try {
    const res = await axios.get(`${API_BASE_URL}/drift-log`);
    return res.data;
  } catch {
    return [];
  }
};
