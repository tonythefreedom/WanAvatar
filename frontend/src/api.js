import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5 minutes timeout for long operations
});

export const healthCheck = () => api.get('/health');

export const getConfig = () => api.get('/config');

export const getLoraAdapters = async (category = null) => {
  const params = category ? { category } : {};
  const response = await api.get('/lora-adapters', { params });
  return response.data;
};

export const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload/image', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const uploadAudio = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload/audio', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const uploadVideo = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload/video', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const startGeneration = async (params) => {
  const response = await api.post('/generate', params);
  return response.data;
};

export const startI2VGeneration = async (params) => {
  const response = await api.post('/generate-i2v', params);
  return response.data;
};

export const getTaskStatus = async (taskId) => {
  const response = await api.get(`/status/${taskId}`);
  return response.data;
};

export const extractAudio = async (videoPath) => {
  const formData = new FormData();
  formData.append('video_path', videoPath);
  const response = await api.post('/extract-audio', formData);
  return response.data;
};

export const separateVocals = async (audioPath) => {
  const formData = new FormData();
  formData.append('audio_path', audioPath);
  const response = await api.post('/separate-vocals', formData);
  return response.data;
};

export const listUploadedImages = async () => {
  const response = await api.get('/uploads/images');
  return response.data;
};

export const listUploadedAudio = async () => {
  const response = await api.get('/uploads/audio');
  return response.data;
};

export const listVideos = async () => {
  const response = await api.get('/videos');
  return response.data;
};

export const deleteVideo = async (filename) => {
  const response = await api.delete(`/videos/${filename}`);
  return response.data;
};

export const getT2iStatus = async () => {
  const response = await api.get('/t2i-status');
  return response.data;
};

export const extractFirstFrame = async (videoPath) => {
  const formData = new FormData();
  formData.append('video_path', videoPath);
  const response = await api.post('/extract-frame', formData);
  return response.data;
};

export const listOutputs = async () => {
  const response = await api.get('/outputs');
  return response.data;
};

export const startFluxGeneration = async (params) => {
  const response = await api.post('/generate-flux', params);
  return response.data;
};

export const deleteOutput = async (filename) => {
  const response = await api.delete(`/outputs/${filename}`);
  return response.data;
};

export default api;
