import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  timeout: 300000, // 5 minutes timeout for long operations
});

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Handle 401 responses (expired/invalid token)
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.dispatchEvent(new Event('auth-logout'));
    }
    return Promise.reject(err);
  }
);

// ─── Auth API ───
export const authLogin = async (email, password) => {
  const response = await api.post('/auth/login', { email, password });
  return response.data;
};

export const authGoogle = async (credential) => {
  const response = await api.post('/auth/google', { credential });
  return response.data;
};

export const authMe = async () => {
  const response = await api.get('/auth/me');
  return response.data;
};

// ─── Admin API ───
export const adminListUsers = async () => {
  const response = await api.get('/admin/users');
  return response.data;
};

export const adminApproveUser = async (id) => {
  const response = await api.post(`/admin/users/${id}/approve`);
  return response.data;
};

export const adminSuspendUser = async (id) => {
  const response = await api.post(`/admin/users/${id}/suspend`);
  return response.data;
};

export const adminActivateUser = async (id) => {
  const response = await api.post(`/admin/users/${id}/activate`);
  return response.data;
};

export const adminDeleteUser = async (id) => {
  const response = await api.delete(`/admin/users/${id}`);
  return response.data;
};

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

export const cancelGeneration = async (taskId) => {
  const response = await api.post(`/cancel/${taskId}`);
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

export const trimVideo = async (videoPath, start, end) => {
  const response = await api.post('/trim-video', { video_path: videoPath, start, end });
  return response.data;
};

export const uploadBackground = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload/background', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
};

export const listBackgrounds = async () => {
  const response = await api.get('/backgrounds');
  return response.data;
};

export const listAvatarGroups = async () => {
  const response = await api.get('/avatars');
  return response.data;
};

export const listAvatarImages = async (group) => {
  const response = await api.get(`/avatars/${group}`);
  return response.data;
};

export const deleteAvatarImage = async (group, filename) => {
  const response = await api.delete(`/avatars/${group}/${filename}`);
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

export const getFashionStyles = async () => {
  const response = await api.get('/fashion-styles');
  return response.data;
};

export const registerAvatar = async (sourcePath, group) => {
  const response = await api.post('/register-avatar', { source_path: sourcePath, group });
  return response.data;
};

export const prepareAvatar = async (sourcePath, group) => {
  const response = await api.post('/prepare-avatar', { source_path: sourcePath, group });
  return response.data;
};

export const uploadToYouTube = async (filename, title, description, hashtags) => {
  const response = await api.post('/upload-youtube', { filename, title, description, hashtags }, { timeout: 600000 });
  return response.data;
};

export default api;
