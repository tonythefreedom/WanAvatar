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

export const startI2VGeneration = async (params) => {
  const response = await api.post('/generate-i2v', params);
  return response.data;
};

export const getTaskStatus = async (taskId) => {
  const response = await api.get(`/status/${taskId}`);
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

export const startWorkflowGeneration = async (params) => {
  const response = await api.post('/workflow/generate', params);
  return response.data;
};

export const getWorkflowStatus = async () => {
  const response = await api.get('/workflow/status');
  return response.data;
};

export const downloadYoutube = async (url) => {
  const response = await api.post('/download-youtube', { url }, { timeout: 600000 });
  return response.data;
};

export const getWorkflows = async () => {
  const response = await api.get('/workflows');
  return response.data;
};

export const prepareWorkflowImages = async (imagePaths) => {
  const response = await api.post('/workflow/prepare-images', { image_paths: imagePaths });
  return response.data;
};

export const generateTTS = async (text, language, speaker) => {
  const response = await api.post('/studio/tts', { text, language, speaker }, { timeout: 120000 });
  return response.data;
};

export const getTTSSpeakers = async () => {
  const response = await api.get('/studio/tts-speakers');
  return response.data;
};

export const sendStudioChat = async (message, history) => {
  const response = await api.post('/studio/chat', { message, history }, { timeout: 180000 });
  return response.data;
};

export default api;
