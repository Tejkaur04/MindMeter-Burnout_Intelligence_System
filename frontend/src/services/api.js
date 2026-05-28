import axios from 'axios';

const API = axios.create({
  baseURL: 'http://127.0.0.1:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

export const apiService = {
  // Submit the 17 metrics from dataset
  predictBurnout: async (payload) => {
    const response = await API.post('/predict', payload);
    return response.data;
  },
  
  // Connect with the supportive AI Coach
  sendMessageToCoach: async (messages) => {
    const response = await API.post('/coach/chat', { messages });
    return response.data;
  },
  
  // Get macro-level platform analytics metrics
  getAnalytics: async () => {
    const response = await API.get('/analytics');
    return response.data;
  },
};