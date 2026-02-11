import { useState, useEffect, useRef, useCallback } from 'react';
import {
  uploadImage,
  uploadAudio,
  uploadVideo,
  startGeneration,
  startI2VGeneration,
  startFluxGeneration,
  getTaskStatus,
  extractAudio,
  separateVocals,
  getConfig,
  getLoraAdapters,
  listVideos,
  deleteVideo,
  deleteOutput,
  listUploadedImages,
  listUploadedAudio,
  getT2iStatus,
  extractFirstFrame,
  listOutputs,
} from './api';
import './App.css';

// Translations
const translations = {
  en: {
    title: 'WanAvatar',
    menuVideoGen: 'Video Gen',
    menuLipsync: 'Voice Lipsync',
    menuGallery: 'Gallery',
    // Lipsync sub-tabs
    subTabGenerate: 'Generate',
    subTabExtract: 'Extract Audio',
    subTabSeparate: 'Vocal Separation',
    // S2V (Lipsync) Generate
    imageLabel: 'Reference Image',
    audioLabel: 'Driving Audio',
    promptLabel: 'Prompt',
    promptDefault: 'A person speaking naturally with subtle expressions, minimal head movement, simple blinking, neutral background',
    negPromptLabel: 'Negative Prompt',
    negPromptDefault: 'ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, three legs, extra limbs',
    resolutionLabel: 'Resolution',
    autoResolution: 'Auto (match image size)',
    imageSize: 'Image Size',
    clipsLabel: 'Number of Clips (0=auto)',
    stepsLabel: 'Inference Steps',
    guidanceLabel: 'Guidance Scale',
    framesLabel: 'Frames per Clip',
    seedLabel: 'Seed (-1=random)',
    offloadLabel: 'Model Offload (saves VRAM)',
    teacacheLabel: 'TeaCache (faster inference)',
    teacacheThreshLabel: 'TeaCache Threshold',
    generateBtn: 'Generate Video',
    generating: 'Generating...',
    videoOutput: 'Generated Video',
    seedOutput: 'Used Seed',
    status: 'Status',
    // Extract / Separate
    videoInput: 'Upload Video',
    extractBtn: 'Extract Audio',
    audioOutput: 'Extracted Audio',
    audioInput: 'Upload Audio',
    separateBtn: 'Separate Vocals',
    vocalsOutput: 'Separated Vocals',
    dropImage: 'Drop image here or click to upload',
    dropAudio: 'Drop audio here or click to upload',
    dropVideo: 'Drop video here or click to upload',
    // Gallery
    galleryTitle: 'Generated Videos',
    galleryEmpty: 'No videos generated yet',
    galleryDelete: 'Delete',
    galleryDeleteConfirm: 'Delete this video?',
    gallerySize: 'Size',
    galleryDate: 'Created',
    galleryRefresh: 'Refresh',
    // File picker
    selectFromUploads: 'Select from uploads',
    noUploads: 'No uploaded files',
    // I2V
    i2vTitle: 'Image to Video (SVI 2.0 Pro)',
    i2vImageLabel: 'Input Image',
    i2vPromptLabel: 'Prompt',
    i2vPromptDefault: 'A cinematic video with natural motion, high quality, smooth movement',
    i2vNegPromptDefault: 'ugly, blurry, low quality, distorted, deformed, static, frozen',
    i2vFrameNumLabel: 'Frame Count',
    i2vShiftLabel: 'Noise Shift',
    i2vGenerateBtn: 'Generate Video',
    i2vGenerating: 'Generating...',
    i2vModelNote: 'First use may take ~1 min for model loading',
    i2vNotAvailable: 'I2V model not available. Download Wan2.2-I2V-14B-A first.',
    // LoRA
    loraTitle: 'LoRA Adapters (High/Low Noise Mix)',
    loraEnabled: 'Enabled',
    loraHighWeight: 'High-Noise Weight',
    loraLowWeight: 'Low-Noise Weight',
    loraCharacter: 'Character',
    loraMotion: 'Motion',
    loraCamera: 'Camera',
    loraTriggerWords: 'Trigger Words',
    loraDescription: 'Description',
    loraInfo: 'Info',
    loraNoAdapters: 'No LoRA adapters available',
    loraHighTip: 'Controls structure, motion, camera (early diffusion steps)',
    loraLowTip: 'Controls appearance, face, texture (late diffusion steps)',
    loraCivitai: 'CivitAI Page',
    loraPreview: 'Preview',
    // Image Gen
    menuImageGen: 'Image Gen',
    imgGenTitle: 'Image Generation (FLUX.2-klein-9B)',
    imgGenNotConfigured: 'T2I model not configured. Upload images manually or add a T2I model later.',
    imgGenUploadTitle: 'Upload / Manage Images',
    imgGenNoImages: 'No images uploaded yet',
    // FLUX
    fluxPromptLabel: 'Prompt',
    fluxPromptDefault: 'K-pop idol, young Korean female, symmetrical face, V-shaped jawline, clear glass skin, double eyelids, trendy idol makeup.\n\nStage lighting, cinematic bokeh, pink and purple neon highlights, professional studio portrait, high-end fashion editorial style.\n\n8k resolution, photorealistic, raw photo, masterwork, intricate details of eyes and hair.',
    fluxStepsLabel: 'Inference Steps',
    fluxGuidanceLabel: 'Guidance Scale',
    fluxUpscaleLabel: 'Upscale x2 (Real-ESRGAN)',
    fluxUpscaleDesc: 'Upscale 720x1280 to 1440x2560',
    fluxGenerateBtn: 'Generate Image',
    fluxGenerating: 'Generating...',
    fluxModelNote: 'FLUX.2-klein-9B: 4-step fast generation. First use requires model download.',
    fluxOutputOriginal: 'Original (720x1280)',
    fluxOutputUpscaled: 'Upscaled (1440x2560)',
    // Gallery
    galleryImages: 'Images',
    galleryVideos: 'Videos',
    // Output picker
    selectFromOutputs: 'Select from generated outputs',
    noOutputs: 'No generated outputs available',
    outputTypeImage: 'Image',
    outputTypeVideo: 'Video (first frame)',
  },
  ko: {
    title: 'WanAvatar',
    menuVideoGen: '비디오 생성',
    menuLipsync: '음성 립싱크',
    menuGallery: '갤러리',
    subTabGenerate: '생성',
    subTabExtract: '오디오 추출',
    subTabSeparate: '보컬 분리',
    imageLabel: '참조 이미지',
    audioLabel: '구동 오디오',
    promptLabel: '프롬프트',
    promptDefault: '자연스럽게 말하는 사람, 미세한 표정, 최소한의 머리 움직임, 단순한 눈 깜빡임, 중립적 배경',
    negPromptLabel: '네거티브 프롬프트',
    negPromptDefault: 'ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, three legs, extra limbs',
    resolutionLabel: '해상도',
    autoResolution: '자동 (이미지 크기에 맞춤)',
    imageSize: '이미지 크기',
    clipsLabel: '클립 수 (0=자동)',
    stepsLabel: '추론 스텝',
    guidanceLabel: '가이던스 스케일',
    framesLabel: '클립당 프레임',
    seedLabel: '시드 (-1=랜덤)',
    offloadLabel: '모델 오프로드 (VRAM 절약)',
    teacacheLabel: 'TeaCache (빠른 추론)',
    teacacheThreshLabel: 'TeaCache 임계값',
    generateBtn: '비디오 생성',
    generating: '생성 중...',
    videoOutput: '생성된 비디오',
    seedOutput: '사용된 시드',
    status: '상태',
    videoInput: '비디오 업로드',
    extractBtn: '오디오 추출',
    audioOutput: '추출된 오디오',
    audioInput: '오디오 업로드',
    separateBtn: '보컬 분리',
    vocalsOutput: '분리된 보컬',
    dropImage: '이미지를 드롭하거나 클릭하여 업로드',
    dropAudio: '오디오를 드롭하거나 클릭하여 업로드',
    dropVideo: '비디오를 드롭하거나 클릭하여 업로드',
    galleryTitle: '생성된 비디오',
    galleryEmpty: '아직 생성된 비디오가 없습니다',
    galleryDelete: '삭제',
    galleryDeleteConfirm: '이 비디오를 삭제하시겠습니까?',
    gallerySize: '크기',
    galleryDate: '생성일',
    galleryRefresh: '새로고침',
    selectFromUploads: '업로드 파일에서 선택',
    noUploads: '업로드된 파일 없음',
    i2vTitle: '이미지-비디오 변환 (SVI 2.0 Pro)',
    i2vImageLabel: '입력 이미지',
    i2vPromptLabel: '프롬프트',
    i2vPromptDefault: '자연스러운 움직임이 있는 시네마틱 영상, 고품질, 부드러운 모션',
    i2vNegPromptDefault: 'ugly, blurry, low quality, distorted, deformed, static, frozen',
    i2vFrameNumLabel: '프레임 수',
    i2vShiftLabel: '노이즈 시프트',
    i2vGenerateBtn: '비디오 생성',
    i2vGenerating: '생성 중...',
    i2vModelNote: '첫 사용 시 모델 로딩에 ~1분 소요',
    i2vNotAvailable: 'I2V 모델이 없습니다. Wan2.2-I2V-14B-A를 먼저 다운로드하세요.',
    // LoRA
    loraTitle: 'LoRA 어댑터 (High/Low 노이즈 믹스)',
    loraEnabled: '활성화',
    loraHighWeight: '하이 노이즈 가중치',
    loraLowWeight: '로우 노이즈 가중치',
    loraCharacter: '캐릭터',
    loraMotion: '모션',
    loraCamera: '카메라',
    loraTriggerWords: '트리거 워드',
    loraDescription: '설명',
    loraInfo: '정보',
    loraNoAdapters: '사용 가능한 LoRA 어댑터 없음',
    loraHighTip: '구조, 동작, 카메라 제어 (초기 확산 단계)',
    loraLowTip: '외관, 얼굴, 텍스처 제어 (후기 확산 단계)',
    loraCivitai: 'CivitAI 페이지',
    loraPreview: '미리보기',
    // Image Gen
    menuImageGen: '이미지 생성',
    imgGenTitle: '이미지 생성 (FLUX.2-klein-9B)',
    imgGenNotConfigured: 'T2I 모델이 구성되지 않았습니다. 이미지를 수동으로 업로드하거나 나중에 T2I 모델을 추가하세요.',
    imgGenUploadTitle: '이미지 업로드 / 관리',
    imgGenNoImages: '아직 업로드된 이미지가 없습니다',
    // FLUX
    fluxPromptLabel: '프롬프트',
    fluxPromptDefault: 'K-pop idol, young Korean female, symmetrical face, V-shaped jawline, clear glass skin, double eyelids, trendy idol makeup.\n\nStage lighting, cinematic bokeh, pink and purple neon highlights, professional studio portrait, high-end fashion editorial style.\n\n8k resolution, photorealistic, raw photo, masterwork, intricate details of eyes and hair.',
    fluxStepsLabel: '추론 스텝',
    fluxGuidanceLabel: '가이던스 스케일',
    fluxUpscaleLabel: '업스케일 x2 (Real-ESRGAN)',
    fluxUpscaleDesc: '720x1280을 1440x2560으로 업스케일',
    fluxGenerateBtn: '이미지 생성',
    fluxGenerating: '생성 중...',
    fluxModelNote: 'FLUX.2-klein-9B: 4스텝 고속 생성. 첫 사용 시 모델 다운로드 필요.',
    fluxOutputOriginal: '원본 (720x1280)',
    fluxOutputUpscaled: '업스케일 (1440x2560)',
    // Gallery
    galleryImages: '이미지',
    galleryVideos: '동영상',
    // Output picker
    selectFromOutputs: '생성된 결과에서 선택',
    noOutputs: '생성된 결과가 없습니다',
    outputTypeImage: '이미지',
    outputTypeVideo: '비디오 (첫 프레임)',
  },
  zh: {
    title: 'WanAvatar',
    menuVideoGen: '视频生成',
    menuLipsync: '语音唇形同步',
    menuGallery: '画廊',
    subTabGenerate: '生成',
    subTabExtract: '提取音频',
    subTabSeparate: '人声分离',
    imageLabel: '参考图片',
    audioLabel: '驱动音频',
    promptLabel: '提示词',
    promptDefault: '一个人自然说话，细微表情，头部动作轻微，简单眨眼，中性背景',
    negPromptLabel: '负面提示词',
    negPromptDefault: '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体灰暗，出现多余的人物',
    resolutionLabel: '分辨率',
    autoResolution: '自动（匹配图片尺寸）',
    imageSize: '图片尺寸',
    clipsLabel: '片段数量（0=自动）',
    stepsLabel: '推理步数',
    guidanceLabel: '引导比例',
    framesLabel: '每片段帧数',
    seedLabel: '种子（-1=随机）',
    offloadLabel: '模型卸载（节省显存）',
    teacacheLabel: 'TeaCache（加速推理）',
    teacacheThreshLabel: 'TeaCache 阈值',
    generateBtn: '生成视频',
    generating: '生成中...',
    videoOutput: '生成的视频',
    seedOutput: '使用的种子',
    status: '状态',
    videoInput: '上传视频',
    extractBtn: '提取音频',
    audioOutput: '提取的音频',
    audioInput: '上传音频',
    separateBtn: '分离人声',
    vocalsOutput: '分离的人声',
    dropImage: '将图片拖放到此处或点击上传',
    dropAudio: '将音频拖放到此处或点击上传',
    dropVideo: '将视频拖放到此处或点击上传',
    galleryTitle: '已生成的视频',
    galleryEmpty: '尚未生成任何视频',
    galleryDelete: '删除',
    galleryDeleteConfirm: '确定删除此视频？',
    gallerySize: '大小',
    galleryDate: '创建时间',
    galleryRefresh: '刷新',
    selectFromUploads: '从已上传文件中选择',
    noUploads: '没有已上传的文件',
    i2vTitle: '图生视频 (SVI 2.0 Pro)',
    i2vImageLabel: '输入图片',
    i2vPromptLabel: '提示词',
    i2vPromptDefault: '电影般的视频，自然运动，高品质，流畅运动',
    i2vNegPromptDefault: 'ugly, blurry, low quality, distorted, deformed, static, frozen',
    i2vFrameNumLabel: '帧数',
    i2vShiftLabel: '噪声偏移',
    i2vGenerateBtn: '生成视频',
    i2vGenerating: '生成中...',
    i2vModelNote: '首次使用需约1分钟加载模型',
    i2vNotAvailable: 'I2V模型不可用，请先下载Wan2.2-I2V-14B-A。',
    // LoRA
    loraTitle: 'LoRA 适配器 (高/低噪声混合)',
    loraEnabled: '启用',
    loraHighWeight: '高噪声权重',
    loraLowWeight: '低噪声权重',
    loraCharacter: '角色',
    loraMotion: '动作',
    loraCamera: '摄影机',
    loraTriggerWords: '触发词',
    loraDescription: '描述',
    loraInfo: '信息',
    loraNoAdapters: '无可用LoRA适配器',
    loraHighTip: '控制结构、运动、镜头（扩散早期步骤）',
    loraLowTip: '控制外观、面部、纹理（扩散后期步骤）',
    loraCivitai: 'CivitAI页面',
    loraPreview: '预览',
    // Image Gen
    menuImageGen: '图像生成',
    imgGenTitle: '图像生成 (FLUX.2-klein-9B)',
    imgGenNotConfigured: 'T2I模型未配置。请手动上传图片或稍后添加T2I模型。',
    imgGenUploadTitle: '上传/管理图片',
    imgGenNoImages: '尚未上传图片',
    // FLUX
    fluxPromptLabel: '提示词',
    fluxPromptDefault: 'K-pop idol, young Korean female, symmetrical face, V-shaped jawline, clear glass skin, double eyelids, trendy idol makeup.\n\nStage lighting, cinematic bokeh, pink and purple neon highlights, professional studio portrait, high-end fashion editorial style.\n\n8k resolution, photorealistic, raw photo, masterwork, intricate details of eyes and hair.',
    fluxStepsLabel: '推理步数',
    fluxGuidanceLabel: '引导比例',
    fluxUpscaleLabel: '放大 x2 (Real-ESRGAN)',
    fluxUpscaleDesc: '将720x1280放大至1440x2560',
    fluxGenerateBtn: '生成图像',
    fluxGenerating: '生成中...',
    fluxModelNote: 'FLUX.2-klein-9B: 4步快速生成。首次使用需下载模型。',
    fluxOutputOriginal: '原始 (720x1280)',
    fluxOutputUpscaled: '放大 (1440x2560)',
    // Gallery
    galleryImages: '图片',
    galleryVideos: '视频',
    // Output picker
    selectFromOutputs: '从生成结果中选择',
    noOutputs: '没有生成结果',
    outputTypeImage: '图片',
    outputTypeVideo: '视频（首帧）',
  },
};

function App() {
  const [lang, setLang] = useState('en');
  const [activeMenu, setActiveMenu] = useState('imagegen');
  const [lipsyncSubTab, setLipsyncSubTab] = useState('generate');
  const [config, setConfig] = useState(null);

  // S2V Generate state
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [imagePath, setImagePath] = useState('');
  const [imageDimensions, setImageDimensions] = useState(null);
  const [autoResolution, setAutoResolution] = useState(true);
  const [audioFile, setAudioFile] = useState(null);
  const [audioPath, setAudioPath] = useState('');
  const [prompt, setPrompt] = useState(translations.en.promptDefault);
  const [negPrompt, setNegPrompt] = useState(translations.en.negPromptDefault);
  const [resolution, setResolution] = useState('720*1280');
  const [numClips, setNumClips] = useState(0);
  const [steps, setSteps] = useState(25);
  const [guidance, setGuidance] = useState(5.5);
  const [frames, setFrames] = useState(80);
  const [seed, setSeed] = useState(-1);
  const [offload, setOffload] = useState(false);
  const [useTeacache, setUseTeacache] = useState(false);
  const [teacacheThresh, setTeacacheThresh] = useState(0.15);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [outputVideo, setOutputVideo] = useState(null);
  const [outputSeed, setOutputSeed] = useState('');

  // Extract tab state
  const [extractVideoFile, setExtractVideoFile] = useState(null);
  const [extractVideoPath, setExtractVideoPath] = useState('');
  const [extractedAudio, setExtractedAudio] = useState(null);
  const [extractStatus, setExtractStatus] = useState('');
  const [isExtracting, setIsExtracting] = useState(false);

  // Separate tab state
  const [separateAudioFile, setSeparateAudioFile] = useState(null);
  const [separateAudioPath, setSeparateAudioPath] = useState('');
  const [separatedVocals, setSeparatedVocals] = useState(null);
  const [separateStatus, setSeparateStatus] = useState('');
  const [isSeparating, setIsSeparating] = useState(false);

  // File picker state
  const [uploadedImages, setUploadedImages] = useState([]);
  const [uploadedAudioList, setUploadedAudioList] = useState([]);
  const [showImagePicker, setShowImagePicker] = useState(false);
  const [showAudioPicker, setShowAudioPicker] = useState(false);

  // Gallery state
  const [videos, setVideos] = useState([]);
  const [galleryLoading, setGalleryLoading] = useState(false);

  // I2V state
  const [i2vImageFile, setI2vImageFile] = useState(null);
  const [i2vImagePreview, setI2vImagePreview] = useState(null);
  const [i2vImagePath, setI2vImagePath] = useState('');
  const [i2vImageDimensions, setI2vImageDimensions] = useState(null);
  const [i2vAutoResolution, setI2vAutoResolution] = useState(true);
  const [i2vPrompt, setI2vPrompt] = useState(translations.en.i2vPromptDefault);
  const [i2vNegPrompt, setI2vNegPrompt] = useState(translations.en.i2vNegPromptDefault);
  const [i2vResolution, setI2vResolution] = useState('720*1280');
  const [i2vFrameNum, setI2vFrameNum] = useState(81);
  const [i2vSteps, setI2vSteps] = useState(40);
  const [i2vGuidance, setI2vGuidance] = useState(5.0);
  const [i2vShift, setI2vShift] = useState(5.0);
  const [i2vSeed, setI2vSeed] = useState(-1);
  const [i2vOffload, setI2vOffload] = useState(false);
  const [i2vIsGenerating, setI2vIsGenerating] = useState(false);
  const [i2vProgress, setI2vProgress] = useState(0);
  const [i2vStatus, setI2vStatus] = useState('');
  const [i2vOutputVideo, setI2vOutputVideo] = useState(null);
  const [i2vOutputSeed, setI2vOutputSeed] = useState('');
  // I2V file picker
  const [showI2vImagePicker, setShowI2vImagePicker] = useState(false);

  // LoRA state
  const [loraAdapters, setLoraAdapters] = useState([]);
  const [loraWeights, setLoraWeights] = useState({}); // {name: {enabled, high_weight, low_weight}}
  const [expandedLora, setExpandedLora] = useState(null); // name of expanded info panel

  // Image Gen state
  const [imgGenImages, setImgGenImages] = useState([]);
  const [t2iAvailable, setT2iAvailable] = useState(false);
  const [t2iMessage, setT2iMessage] = useState('');

  // FLUX generation state
  const [fluxPrompt, setFluxPrompt] = useState('K-pop idol, young Korean female, symmetrical face, V-shaped jawline, clear glass skin, double eyelids, trendy idol makeup.\n\nStage lighting, cinematic bokeh, pink and purple neon highlights, professional studio portrait, high-end fashion editorial style.\n\n8k resolution, photorealistic, raw photo, masterwork, intricate details of eyes and hair.');
  const [fluxSteps, setFluxSteps] = useState(4);
  const [fluxGuidance, setFluxGuidance] = useState(1.0);
  const [fluxSeed, setFluxSeed] = useState(-1);
  const [fluxUpscale, setFluxUpscale] = useState(false);
  const [fluxIsGenerating, setFluxIsGenerating] = useState(false);
  const [fluxProgress, setFluxProgress] = useState(0);
  const [fluxStatus, setFluxStatus] = useState('');
  const [fluxOutputImage, setFluxOutputImage] = useState(null);
  const [fluxOutputUpscaled, setFluxOutputUpscaled] = useState(null);
  const [fluxOutputSeed, setFluxOutputSeed] = useState('');

  // Gallery tab state
  const [galleryTab, setGalleryTab] = useState('videos');
  const [galleryImages, setGalleryImages] = useState([]);

  // Image-category LoRA state (separate from mov LoRAs)
  const [imgLoraAdapters, setImgLoraAdapters] = useState([]);
  const [imgLoraWeights, setImgLoraWeights] = useState({});
  const [expandedImgLora, setExpandedImgLora] = useState(null);

  // Lipsync output picker state
  const [showOutputPicker, setShowOutputPicker] = useState(false);
  const [generatedOutputs, setGeneratedOutputs] = useState([]);
  const [isExtractingFrame, setIsExtractingFrame] = useState(false);

  const t = useCallback((key) => translations[lang][key] || key, [lang]);

  // Load config & LoRA adapters
  useEffect(() => {
    getConfig().then(res => setConfig(res.data)).catch(console.error);

    // Load mov-category LoRAs (for Video Gen page)
    getLoraAdapters('mov').then(data => {
      setLoraAdapters(data.adapters || []);
      const defaults = {};
      (data.adapters || []).forEach(a => {
        defaults[a.name] = {
          enabled: a.available,
          high_weight: a.default_high_weight,
          low_weight: a.default_low_weight,
        };
      });
      setLoraWeights(defaults);
    }).catch(console.error);

    // Load img-category LoRAs (for Image Gen page)
    getLoraAdapters('img').then(data => {
      setImgLoraAdapters(data.adapters || []);
      const defaults = {};
      (data.adapters || []).forEach(a => {
        defaults[a.name] = {
          enabled: a.available,
          high_weight: a.default_high_weight,
          low_weight: a.default_low_weight,
        };
      });
      setImgLoraWeights(defaults);
    }).catch(console.error);

    // Check T2I availability
    getT2iStatus().then(data => {
      setT2iAvailable(data.available);
      setT2iMessage(data.message);
    }).catch(console.error);
  }, []);

  // Update prompts when language changes
  useEffect(() => {
    setPrompt(translations[lang].promptDefault);
    setNegPrompt(translations[lang].negPromptDefault);
    setI2vPrompt(translations[lang].i2vPromptDefault);
    setI2vNegPrompt(translations[lang].i2vNegPromptDefault);
  }, [lang]);

  // === S2V Handlers ===
  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
    try {
      const result = await uploadImage(file);
      setImagePath(result.path);
      if (result.width && result.height) {
        setImageDimensions({ width: result.width, height: result.height });
        if (autoResolution) setResolution(`${result.height}*${result.width}`);
      }
    } catch (err) {
      setStatus(`Error uploading image: ${err.message}`);
    }
  };

  const handleAudioUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setAudioFile(file);
    try {
      const result = await uploadAudio(file);
      setAudioPath(result.path);
    } catch (err) {
      setStatus(`Error uploading audio: ${err.message}`);
    }
  };

  const handleGenerate = async () => {
    if (!imagePath || !audioPath) { setStatus('Please upload both image and audio'); return; }
    setIsGenerating(true); setProgress(0); setStatus('Starting generation...'); setOutputVideo(null);
    try {
      // Build LoRA weights for request
      const activeLoras = Object.entries(loraWeights)
        .filter(([, w]) => w.enabled && (w.high_weight > 0 || w.low_weight > 0))
        .map(([name, w]) => ({ name, high_weight: w.high_weight, low_weight: w.low_weight }));

      const { task_id } = await startGeneration({
        image_path: imagePath, audio_path: audioPath, prompt, negative_prompt: negPrompt,
        resolution, num_clips: numClips, inference_steps: steps, guidance_scale: guidance,
        infer_frames: frames, seed, offload_model: offload,
        use_teacache: useTeacache, teacache_thresh: teacacheThresh,
        lora_weights: activeLoras.length > 0 ? activeLoras : null,
      });
      const pollInterval = setInterval(async () => {
        try {
          const ts = await getTaskStatus(task_id);
          setProgress(ts.progress * 100); setStatus(ts.message);
          if (ts.status === 'completed') {
            clearInterval(pollInterval); setIsGenerating(false);
            setOutputVideo(ts.output_path); setOutputSeed(ts.seed?.toString() || '');
          } else if (ts.status === 'failed') {
            clearInterval(pollInterval); setIsGenerating(false); setStatus(`Error: ${ts.message}`);
          }
        } catch (err) { clearInterval(pollInterval); setIsGenerating(false); setStatus(`Error: ${err.message}`); }
      }, 2000);
    } catch (err) { setIsGenerating(false); setStatus(`Error: ${err.message}`); }
  };

  // === I2V Handlers ===
  const handleI2vImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setI2vImageFile(file);
    setI2vImagePreview(URL.createObjectURL(file));
    try {
      const result = await uploadImage(file);
      setI2vImagePath(result.path);
      if (result.width && result.height) {
        setI2vImageDimensions({ width: result.width, height: result.height });
        if (i2vAutoResolution) setI2vResolution(`${result.height}*${result.width}`);
      }
    } catch (err) {
      setI2vStatus(`Error uploading image: ${err.message}`);
    }
  };

  const handleI2vGenerate = async () => {
    if (!i2vImagePath) { setI2vStatus('Please upload an image'); return; }
    setI2vIsGenerating(true); setI2vProgress(0); setI2vStatus('Starting I2V generation...'); setI2vOutputVideo(null);
    try {
      // Build LoRA weights for request
      const activeLoras = Object.entries(loraWeights)
        .filter(([, w]) => w.enabled && (w.high_weight > 0 || w.low_weight > 0))
        .map(([name, w]) => ({ name, high_weight: w.high_weight, low_weight: w.low_weight }));

      const { task_id } = await startI2VGeneration({
        image_path: i2vImagePath, prompt: i2vPrompt, negative_prompt: i2vNegPrompt,
        resolution: i2vResolution, frame_num: i2vFrameNum, inference_steps: i2vSteps,
        guidance_scale: i2vGuidance, shift: i2vShift, seed: i2vSeed, offload_model: i2vOffload,
        lora_weights: activeLoras.length > 0 ? activeLoras : null,
      });
      const pollInterval = setInterval(async () => {
        try {
          const ts = await getTaskStatus(task_id);
          setI2vProgress(ts.progress * 100); setI2vStatus(ts.message);
          if (ts.status === 'completed') {
            clearInterval(pollInterval); setI2vIsGenerating(false);
            setI2vOutputVideo(ts.output_path); setI2vOutputSeed(ts.seed?.toString() || '');
          } else if (ts.status === 'failed') {
            clearInterval(pollInterval); setI2vIsGenerating(false); setI2vStatus(`Error: ${ts.message}`);
          }
        } catch (err) { clearInterval(pollInterval); setI2vIsGenerating(false); setI2vStatus(`Error: ${err.message}`); }
      }, 2000);
    } catch (err) { setI2vIsGenerating(false); setI2vStatus(`Error: ${err.message}`); }
  };

  // === Extract / Separate Handlers ===
  const handleExtractVideoUpload = async (e) => {
    const file = e.target.files[0]; if (!file) return;
    setExtractVideoFile(file);
    try { const result = await uploadVideo(file); setExtractVideoPath(result.path); }
    catch (err) { setExtractStatus(`Error: ${err.message}`); }
  };

  const handleExtract = async () => {
    if (!extractVideoPath) { setExtractStatus('Please upload a video'); return; }
    setIsExtracting(true); setExtractStatus('Extracting audio...'); setExtractedAudio(null);
    try {
      const result = await extractAudio(extractVideoPath);
      setExtractedAudio(result.url); setExtractStatus('Audio extracted successfully!');
    } catch (err) { setExtractStatus(`Error: ${err.message}`); }
    finally { setIsExtracting(false); }
  };

  const handleSeparateAudioUpload = async (e) => {
    const file = e.target.files[0]; if (!file) return;
    setSeparateAudioFile(file);
    try { const result = await uploadAudio(file); setSeparateAudioPath(result.path); }
    catch (err) { setSeparateStatus(`Error: ${err.message}`); }
  };

  const handleSeparate = async () => {
    if (!separateAudioPath) { setSeparateStatus('Please upload audio'); return; }
    setIsSeparating(true); setSeparateStatus('Separating vocals...'); setSeparatedVocals(null);
    try {
      const result = await separateVocals(separateAudioPath);
      setSeparatedVocals(result.url); setSeparateStatus('Vocals separated successfully!');
    } catch (err) { setSeparateStatus(`Error: ${err.message}`); }
    finally { setIsSeparating(false); }
  };

  // === Gallery Handlers ===
  const fetchGallery = useCallback(async () => {
    setGalleryLoading(true);
    try {
      const data = await listOutputs();
      const outputs = data.outputs || [];
      setGalleryImages(outputs.filter(o => o.type === 'image'));
      setVideos(outputs.filter(o => o.type === 'video'));
    }
    catch (err) { console.error('Failed to fetch gallery:', err); }
    finally { setGalleryLoading(false); }
  }, []);

  const handleDeleteOutput = async (filename) => {
    if (!window.confirm(t('galleryDeleteConfirm'))) return;
    try {
      await deleteOutput(filename);
      setGalleryImages(prev => prev.filter(o => o.filename !== filename));
      setVideos(prev => prev.filter(v => v.filename !== filename));
    } catch (err) { console.error('Failed to delete:', err); }
  };

  useEffect(() => { if (activeMenu === 'gallery') fetchGallery(); }, [activeMenu, fetchGallery]);

  // === File Picker Handlers ===
  const toggleImagePicker = async () => {
    if (!showImagePicker) {
      try { const data = await listUploadedImages(); setUploadedImages(data.images || []); }
      catch (err) { console.error(err); }
    }
    setShowImagePicker(!showImagePicker);
  };

  const toggleAudioPicker = async () => {
    if (!showAudioPicker) {
      try { const data = await listUploadedAudio(); setUploadedAudioList(data.audio || []); }
      catch (err) { console.error(err); }
    }
    setShowAudioPicker(!showAudioPicker);
  };

  const selectUploadedImage = (img) => {
    setImagePath(img.path); setImagePreview(img.url); setImageFile({ name: img.filename });
    if (img.width && img.height) {
      setImageDimensions({ width: img.width, height: img.height });
      if (autoResolution) setResolution(`${img.height}*${img.width}`);
    }
    setShowImagePicker(false);
  };

  const selectUploadedAudio = (audio) => {
    setAudioPath(audio.path); setAudioFile({ name: audio.filename }); setShowAudioPicker(false);
  };

  // I2V file picker
  const toggleI2vImagePicker = async () => {
    if (!showI2vImagePicker) {
      try { const data = await listUploadedImages(); setUploadedImages(data.images || []); }
      catch (err) { console.error(err); }
    }
    setShowI2vImagePicker(!showI2vImagePicker);
  };

  const selectI2vUploadedImage = (img) => {
    setI2vImagePath(img.path); setI2vImagePreview(img.url); setI2vImageFile({ name: img.filename });
    if (img.width && img.height) {
      setI2vImageDimensions({ width: img.width, height: img.height });
      if (i2vAutoResolution) setI2vResolution(`${img.height}*${img.width}`);
    }
    setShowI2vImagePicker(false);
  };

  // === Image Gen Handlers ===
  const handleImgGenUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      await uploadImage(file);
      const data = await listUploadedImages();
      setImgGenImages(data.images || []);
    } catch (err) {
      console.error('Image upload error:', err);
    }
  };

  useEffect(() => {
    if (activeMenu === 'imagegen') {
      listUploadedImages().then(data => setImgGenImages(data.images || [])).catch(console.error);
    }
  }, [activeMenu]);

  // Update FLUX prompt when language changes
  useEffect(() => {
    setFluxPrompt(translations[lang].fluxPromptDefault);
  }, [lang]);

  // === FLUX Generation Handler ===
  const handleFluxGenerate = async () => {
    setFluxIsGenerating(true); setFluxProgress(0); setFluxStatus('Starting FLUX generation...');
    setFluxOutputImage(null); setFluxOutputUpscaled(null);
    try {
      // Build img LoRA weights
      const activeLoras = Object.entries(imgLoraWeights)
        .filter(([, w]) => w.enabled && (w.high_weight > 0 || w.low_weight > 0))
        .map(([name, w]) => ({ name, weight: Math.max(w.high_weight, w.low_weight) }));

      const { task_id } = await startFluxGeneration({
        prompt: fluxPrompt,
        num_inference_steps: fluxSteps,
        guidance_scale: fluxGuidance,
        seed: fluxSeed,
        upscale: fluxUpscale,
        lora_weights: activeLoras.length > 0 ? activeLoras : null,
      });
      const pollInterval = setInterval(async () => {
        try {
          const ts = await getTaskStatus(task_id);
          setFluxProgress(ts.progress * 100); setFluxStatus(ts.message);
          if (ts.status === 'completed') {
            clearInterval(pollInterval); setFluxIsGenerating(false);
            setFluxOutputImage(ts.output_path);
            if (ts.upscaled_path) setFluxOutputUpscaled(ts.upscaled_path);
            setFluxOutputSeed(ts.seed?.toString() || '');
          } else if (ts.status === 'failed') {
            clearInterval(pollInterval); setFluxIsGenerating(false);
            setFluxStatus(`Error: ${ts.message}`);
          }
        } catch (err) { clearInterval(pollInterval); setFluxIsGenerating(false); setFluxStatus(`Error: ${err.message}`); }
      }, 2000);
    } catch (err) { setFluxIsGenerating(false); setFluxStatus(`Error: ${err.message}`); }
  };

  // === Lipsync Output Picker Handlers ===
  const toggleOutputPicker = async () => {
    if (!showOutputPicker) {
      try {
        const data = await listOutputs();
        setGeneratedOutputs(data.outputs || []);
      } catch (err) { console.error(err); }
    }
    setShowOutputPicker(!showOutputPicker);
  };

  const selectOutputAsReference = async (output) => {
    setShowOutputPicker(false);
    if (output.type === 'image') {
      setImagePath(output.path);
      setImagePreview(output.url);
      setImageFile({ name: output.filename });
      if (output.width && output.height) {
        setImageDimensions({ width: output.width, height: output.height });
        if (autoResolution) setResolution(`${output.height}*${output.width}`);
      }
    } else if (output.type === 'video') {
      setIsExtractingFrame(true);
      setStatus('Extracting first frame from video...');
      try {
        const frameData = await extractFirstFrame(output.path);
        setImagePath(frameData.path);
        setImagePreview(frameData.url);
        setImageFile({ name: frameData.url.split('/').pop() });
        if (frameData.width && frameData.height) {
          setImageDimensions({ width: frameData.width, height: frameData.height });
          if (autoResolution) setResolution(`${frameData.height}*${frameData.width}`);
        }
        setStatus('Frame extracted successfully');
      } catch (err) {
        setStatus(`Error extracting frame: ${err.message}`);
      } finally {
        setIsExtractingFrame(false);
      }
    }
  };

  return (
    <div className="app-layout">
      {/* Top Header */}
      <header className="top-header">
        <h1 className="logo">{t('title')}</h1>
        <div className="language-selector">
          <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>EN</button>
          <button className={lang === 'ko' ? 'active' : ''} onClick={() => setLang('ko')}>KO</button>
          <button className={lang === 'zh' ? 'active' : ''} onClick={() => setLang('zh')}>ZH</button>
        </div>
      </header>

      <div className="app-body">
        {/* Sidebar */}
        <nav className="sidebar">
          <div
            className={`sidebar-item${activeMenu === 'imagegen' ? ' active' : ''}`}
            onClick={() => setActiveMenu('imagegen')}
          >
            <span className="sidebar-icon">&#128444;</span>
            {t('menuImageGen')}
          </div>
          <div
            className={`sidebar-item${activeMenu === 'videogen' ? ' active' : ''}`}
            onClick={() => setActiveMenu('videogen')}
          >
            <span className="sidebar-icon">&#9654;</span>
            {t('menuVideoGen')}
          </div>
          <div
            className={`sidebar-item${activeMenu === 'lipsync' ? ' active' : ''}`}
            onClick={() => setActiveMenu('lipsync')}
          >
            <span className="sidebar-icon">&#127908;</span>
            {t('menuLipsync')}
          </div>
          <div
            className={`sidebar-item${activeMenu === 'gallery' ? ' active' : ''}`}
            onClick={() => setActiveMenu('gallery')}
          >
            <span className="sidebar-icon">&#128247;</span>
            {t('menuGallery')}
          </div>
        </nav>

        {/* Main Content */}
        <main className="main-content">

          {/* ============ IMAGE GEN ============ */}
          {activeMenu === 'imagegen' && (
            <div className="page-content">
              <h2 className="page-title">{t('imgGenTitle')}</h2>
              <p className="model-note">{t('fluxModelNote')}</p>

              <div className="two-column">
                {/* Left: Prompt + Settings + LoRA */}
                <div className="column">
                  <div className="card">
                    <h3>{t('fluxPromptLabel')}</h3>
                    <div className="form-group">
                      <textarea value={fluxPrompt} onChange={(e) => setFluxPrompt(e.target.value)} rows={4} />
                    </div>

                    <div className="form-row">
                      <div className="form-group">
                        <label>{t('fluxStepsLabel')}: {fluxSteps}</label>
                        <input type="range" min={1} max={12} step={1} value={fluxSteps} onChange={(e) => setFluxSteps(parseInt(e.target.value))} />
                      </div>
                      <div className="form-group">
                        <label>{t('fluxGuidanceLabel')}: {fluxGuidance}</label>
                        <input type="range" min={1} max={10} step={0.5} value={fluxGuidance} onChange={(e) => setFluxGuidance(parseFloat(e.target.value))} />
                      </div>
                    </div>

                    <div className="form-row">
                      <div className="form-group">
                        <label>{t('seedLabel')}</label>
                        <input type="number" value={fluxSeed} onChange={(e) => setFluxSeed(parseInt(e.target.value))} />
                      </div>
                      <div className="form-group checkbox">
                        <label>
                          <input type="checkbox" checked={fluxUpscale} onChange={(e) => setFluxUpscale(e.target.checked)} />
                          {t('fluxUpscaleLabel')}
                        </label>
                        <span className="image-size-info">{t('fluxUpscaleDesc')}</span>
                      </div>
                    </div>

                    <button className="btn primary" onClick={handleFluxGenerate} disabled={fluxIsGenerating || !fluxPrompt.trim()}>
                      {fluxIsGenerating ? t('fluxGenerating') : t('fluxGenerateBtn')}
                    </button>
                  </div>

                  {/* Upload images manually */}
                  <div className="card">
                    <h3>{t('imgGenUploadTitle')}</h3>
                    <div className="file-upload">
                      <input type="file" accept="image/*" onChange={handleImgGenUpload} id="imggen-upload" />
                      <label htmlFor="imggen-upload" className="upload-area small">
                        <span>{t('dropImage')}</span>
                      </label>
                    </div>
                    {imgGenImages.length > 0 && (
                      <div className="picker-list" style={{ maxHeight: '200px' }}>
                        {imgGenImages.map((img) => (
                          <div key={img.filename} className="picker-item">
                            <img src={img.url} alt={img.filename} className="picker-thumb" />
                            <div className="picker-info">
                              <span className="picker-name">{img.filename}</span>
                              <span className="picker-meta">{img.width}x{img.height}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Image LoRA panel */}
                  {imgLoraAdapters.length > 0 && (
                    <div className="card lora-card">
                      <h3>{t('loraTitle')}</h3>
                      {imgLoraAdapters.map(adapter => {
                        const w = imgLoraWeights[adapter.name] || { enabled: false, high_weight: 0, low_weight: 0 };
                        const isExpanded = expandedImgLora === adapter.name;
                        const typeLabel = adapter.type === 'character' ? t('loraCharacter') : adapter.type === 'camera' ? t('loraCamera') : t('loraMotion');
                        return (
                          <div key={adapter.name} className={`lora-adapter${w.enabled ? ' enabled' : ''}`}>
                            <div className="lora-adapter-header">
                              <label className="lora-toggle">
                                <input type="checkbox" checked={w.enabled} disabled={!adapter.available}
                                  onChange={(e) => setImgLoraWeights(prev => ({
                                    ...prev, [adapter.name]: { ...prev[adapter.name], enabled: e.target.checked }
                                  }))} />
                                <span className="lora-name">{adapter.name}</span>
                                <span className={`lora-type-badge ${adapter.type}`}>{typeLabel}</span>
                              </label>
                              <button className="btn secondary small lora-info-btn"
                                onClick={() => setExpandedImgLora(isExpanded ? null : adapter.name)}>
                                {t('loraInfo')} {isExpanded ? '\u25B2' : '\u25BC'}
                              </button>
                            </div>
                            {isExpanded && (
                              <div className="lora-info-panel">
                                <p className="lora-desc">{adapter.description}</p>
                                {adapter.trigger_words?.length > 0 && (
                                  <div className="lora-trigger">
                                    <strong>{t('loraTriggerWords')}:</strong>
                                    <div className="lora-tags">
                                      {adapter.trigger_words.map(tw => <span key={tw} className="lora-tag">{tw}</span>)}
                                    </div>
                                  </div>
                                )}
                                {adapter.civitai_url && (
                                  <a href={adapter.civitai_url} target="_blank" rel="noopener noreferrer" className="lora-civitai-link">
                                    {t('loraCivitai')} &rarr;
                                  </a>
                                )}
                              </div>
                            )}
                            {w.enabled && (
                              <div className="lora-weights">
                                <div className="form-group">
                                  <label>{t('loraHighWeight')}: {w.high_weight.toFixed(2)}</label>
                                  <input type="range" min={0} max={1.5} step={0.05} value={w.high_weight}
                                    onChange={(e) => setImgLoraWeights(prev => ({
                                      ...prev, [adapter.name]: { ...prev[adapter.name], high_weight: parseFloat(e.target.value) }
                                    }))} />
                                </div>
                                <div className="form-group">
                                  <label>{t('loraLowWeight')}: {w.low_weight.toFixed(2)}</label>
                                  <input type="range" min={0} max={1.5} step={0.05} value={w.low_weight}
                                    onChange={(e) => setImgLoraWeights(prev => ({
                                      ...prev, [adapter.name]: { ...prev[adapter.name], low_weight: parseFloat(e.target.value) }
                                    }))} />
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* Right: Output */}
                <div className="column">
                  <div className="card">
                    <h3>Output</h3>
                    {fluxIsGenerating && (
                      <div className="progress-container">
                        <div className="progress-bar"><div className="progress-fill" style={{ width: `${fluxProgress}%` }} /></div>
                        <span className="progress-text">{Math.round(fluxProgress)}%</span>
                      </div>
                    )}
                    {fluxOutputImage && (
                      <div className="flux-output">
                        <div className="flux-output-item">
                          <label>{t('fluxOutputOriginal')}</label>
                          <img src={fluxOutputImage} alt="Generated" className="flux-output-img" />
                        </div>
                        {fluxOutputUpscaled && (
                          <div className="flux-output-item">
                            <label>{t('fluxOutputUpscaled')}</label>
                            <img src={fluxOutputUpscaled} alt="Upscaled" className="flux-output-img" />
                          </div>
                        )}
                      </div>
                    )}
                    {!fluxOutputImage && !fluxIsGenerating && (
                      <div className="placeholder"><span>FLUX.2-klein-9B Output</span></div>
                    )}
                    {fluxOutputSeed && <div className="form-group"><label>{t('seedOutput')}</label><input type="text" value={fluxOutputSeed} readOnly /></div>}
                    <div className="status-box"><label>{t('status')}</label><p>{fluxStatus || 'Ready'}</p></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ============ VIDEO GEN (I2V) ============ */}
          {activeMenu === 'videogen' && (
            <div className="page-content">
              <h2 className="page-title">{t('i2vTitle')}</h2>
              {config && !config.i2v_available && (
                <div className="alert">{t('i2vNotAvailable')}</div>
              )}
              <p className="model-note">{t('i2vModelNote')}</p>

              <div className="two-column">
                {/* Left — Inputs */}
                <div className="column">
                  <div className="card">
                    <h3>Input</h3>
                    <div className="form-group">
                      <label>{t('i2vImageLabel')}</label>
                      <div className="file-upload">
                        <input type="file" accept="image/*" onChange={handleI2vImageUpload} id="i2v-image-upload" />
                        <label htmlFor="i2v-image-upload" className="upload-area">
                          {i2vImagePreview ? <img src={i2vImagePreview} alt="Preview" className="preview-image" /> : <span>{t('dropImage')}</span>}
                        </label>
                      </div>
                    </div>
                    <button className="btn secondary small" onClick={toggleI2vImagePicker}>
                      {t('selectFromUploads')}
                    </button>
                    {showI2vImagePicker && (
                      <div className="picker-list">
                        {uploadedImages.length === 0 ? <p className="picker-empty">{t('noUploads')}</p> : uploadedImages.map((img) => (
                          <div key={img.filename} className={`picker-item${i2vImagePath === img.path ? ' selected' : ''}`} onClick={() => selectI2vUploadedImage(img)}>
                            <img src={img.url} alt={img.filename} className="picker-thumb" />
                            <div className="picker-info">
                              <span className="picker-name">{img.filename}</span>
                              <span className="picker-meta">{img.width}x{img.height} / {(img.size / 1024 / 1024).toFixed(1)}MB</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  <div className="card">
                    <h3>Settings</h3>
                    <div className="form-group">
                      <label>{t('i2vPromptLabel')}</label>
                      <textarea value={i2vPrompt} onChange={(e) => setI2vPrompt(e.target.value)} rows={3} />
                    </div>
                    <div className="form-group">
                      <label>{t('negPromptLabel')}</label>
                      <textarea value={i2vNegPrompt} onChange={(e) => setI2vNegPrompt(e.target.value)} rows={2} />
                    </div>

                    <div className="form-group checkbox">
                      <label>
                        <input type="checkbox" checked={i2vAutoResolution} onChange={(e) => {
                          setI2vAutoResolution(e.target.checked);
                          if (e.target.checked && i2vImageDimensions) setI2vResolution(`${i2vImageDimensions.height}*${i2vImageDimensions.width}`);
                        }} />
                        {t('autoResolution')}
                      </label>
                      {i2vImageDimensions && <span className="image-size-info">{t('imageSize')}: {i2vImageDimensions.width} x {i2vImageDimensions.height}</span>}
                    </div>

                    <div className="form-row">
                      <div className="form-group">
                        <label>{t('resolutionLabel')}: {i2vResolution}</label>
                        <input type="text" value={i2vResolution} onChange={(e) => setI2vResolution(e.target.value)} placeholder="height*width" />
                      </div>
                      <div className="form-group">
                        <label>{t('i2vFrameNumLabel')}: {i2vFrameNum}</label>
                        <input type="range" min={17} max={121} step={4} value={i2vFrameNum} onChange={(e) => setI2vFrameNum(parseInt(e.target.value))} />
                      </div>
                    </div>

                    <div className="form-row">
                      <div className="form-group">
                        <label>{t('stepsLabel')}: {i2vSteps}</label>
                        <input type="range" min={5} max={50} step={1} value={i2vSteps} onChange={(e) => setI2vSteps(parseInt(e.target.value))} />
                      </div>
                      <div className="form-group">
                        <label>{t('guidanceLabel')}: {i2vGuidance}</label>
                        <input type="range" min={1} max={10} step={0.5} value={i2vGuidance} onChange={(e) => setI2vGuidance(parseFloat(e.target.value))} />
                      </div>
                    </div>

                    <div className="form-row">
                      <div className="form-group">
                        <label>{t('i2vShiftLabel')}: {i2vShift}</label>
                        <input type="range" min={1} max={10} step={0.5} value={i2vShift} onChange={(e) => setI2vShift(parseFloat(e.target.value))} />
                      </div>
                      <div className="form-group">
                        <label>{t('seedLabel')}</label>
                        <input type="number" value={i2vSeed} onChange={(e) => setI2vSeed(parseInt(e.target.value))} />
                      </div>
                    </div>

                    <div className="form-group checkbox">
                      <label><input type="checkbox" checked={i2vOffload} onChange={(e) => setI2vOffload(e.target.checked)} /> {t('offloadLabel')}</label>
                    </div>

                    <button className="btn primary" onClick={handleI2vGenerate} disabled={i2vIsGenerating || !i2vImagePath}>
                      {i2vIsGenerating ? t('i2vGenerating') : t('i2vGenerateBtn')}
                    </button>
                  </div>

                  {/* LoRA Adapters Card */}
                  {loraAdapters.length > 0 && (
                    <div className="card lora-card">
                      <h3>{t('loraTitle')}</h3>
                      <div className="lora-strategy-info">
                        <p><strong>High-Noise:</strong> {t('loraHighTip')}</p>
                        <p><strong>Low-Noise:</strong> {t('loraLowTip')}</p>
                      </div>
                      {loraAdapters.map(adapter => {
                        const w = loraWeights[adapter.name] || { enabled: false, high_weight: 0, low_weight: 0 };
                        const isExpanded = expandedLora === adapter.name;
                        const typeLabel = adapter.type === 'character' ? t('loraCharacter') : adapter.type === 'camera' ? t('loraCamera') : t('loraMotion');
                        return (
                          <div key={adapter.name} className={`lora-adapter${w.enabled ? ' enabled' : ''}`}>
                            <div className="lora-adapter-header">
                              <label className="lora-toggle">
                                <input type="checkbox" checked={w.enabled} disabled={!adapter.available}
                                  onChange={(e) => setLoraWeights(prev => ({
                                    ...prev, [adapter.name]: { ...prev[adapter.name], enabled: e.target.checked }
                                  }))} />
                                <span className="lora-name">{adapter.name}</span>
                                <span className={`lora-type-badge ${adapter.type}`}>{typeLabel}</span>
                              </label>
                              <button className="btn secondary small lora-info-btn"
                                onClick={() => setExpandedLora(isExpanded ? null : adapter.name)}>
                                {t('loraInfo')} {isExpanded ? '\u25B2' : '\u25BC'}
                              </button>
                            </div>

                            {isExpanded && (
                              <div className="lora-info-panel">
                                <p className="lora-desc">{adapter.description}</p>
                                {adapter.trigger_words?.length > 0 && (
                                  <div className="lora-trigger">
                                    <strong>{t('loraTriggerWords')}:</strong>
                                    <div className="lora-tags">
                                      {adapter.trigger_words.map(tw => (
                                        <span key={tw} className="lora-tag">{tw}</span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                                {adapter.preview_urls?.length > 0 && (
                                  <div className="lora-previews">
                                    <strong>{t('loraPreview')}:</strong>
                                    <div className="lora-preview-grid">
                                      {adapter.preview_urls.map((url, idx) => (
                                        url.endsWith('.mp4') ?
                                          <video key={idx} src={url} controls preload="metadata" className="lora-preview-media" /> :
                                          <img key={idx} src={url} alt={`${adapter.name} preview ${idx+1}`} className="lora-preview-media" />
                                      ))}
                                    </div>
                                  </div>
                                )}
                                {adapter.civitai_url && (
                                  <a href={adapter.civitai_url} target="_blank" rel="noopener noreferrer" className="lora-civitai-link">
                                    {t('loraCivitai')} &rarr;
                                  </a>
                                )}
                              </div>
                            )}

                            {w.enabled && (
                              <div className="lora-weights">
                                <div className="form-group">
                                  <label>{t('loraHighWeight')}: {w.high_weight.toFixed(2)}</label>
                                  <input type="range" min={0} max={1.5} step={0.05} value={w.high_weight}
                                    onChange={(e) => setLoraWeights(prev => ({
                                      ...prev, [adapter.name]: { ...prev[adapter.name], high_weight: parseFloat(e.target.value) }
                                    }))} />
                                </div>
                                <div className="form-group">
                                  <label>{t('loraLowWeight')}: {w.low_weight.toFixed(2)}</label>
                                  <input type="range" min={0} max={1.5} step={0.05} value={w.low_weight}
                                    onChange={(e) => setLoraWeights(prev => ({
                                      ...prev, [adapter.name]: { ...prev[adapter.name], low_weight: parseFloat(e.target.value) }
                                    }))} />
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* Right — Output */}
                <div className="column">
                  <div className="card">
                    <h3>Output</h3>
                    {i2vIsGenerating && (
                      <div className="progress-container">
                        <div className="progress-bar"><div className="progress-fill" style={{ width: `${i2vProgress}%` }} /></div>
                        <span className="progress-text">{Math.round(i2vProgress)}%</span>
                      </div>
                    )}
                    <div className="video-container">
                      {i2vOutputVideo ? <video controls src={i2vOutputVideo} /> : <div className="placeholder"><span>{t('videoOutput')}</span></div>}
                    </div>
                    {i2vOutputSeed && <div className="form-group"><label>{t('seedOutput')}</label><input type="text" value={i2vOutputSeed} readOnly /></div>}
                    <div className="status-box"><label>{t('status')}</label><p>{i2vStatus || 'Ready'}</p></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ============ VOICE LIPSYNC (S2V) ============ */}
          {activeMenu === 'lipsync' && (
            <div className="page-content">
              {/* Sub-tabs */}
              <div className="sub-tabs">
                <button className={lipsyncSubTab === 'generate' ? 'active' : ''} onClick={() => setLipsyncSubTab('generate')}>{t('subTabGenerate')}</button>
                <button className={lipsyncSubTab === 'extract' ? 'active' : ''} onClick={() => setLipsyncSubTab('extract')}>{t('subTabExtract')}</button>
                <button className={lipsyncSubTab === 'separate' ? 'active' : ''} onClick={() => setLipsyncSubTab('separate')}>{t('subTabSeparate')}</button>
              </div>

              {/* S2V Generate */}
              {lipsyncSubTab === 'generate' && (
                <div className="two-column">
                  <div className="column">
                    <div className="card">
                      <h3>Input</h3>
                      <div className="form-group">
                        <label>{t('imageLabel')}</label>
                        <div className="file-upload">
                          <input type="file" accept="image/*" onChange={handleImageUpload} id="image-upload" />
                          <label htmlFor="image-upload" className="upload-area">
                            {imagePreview ? <img src={imagePreview} alt="Preview" className="preview-image" /> : <span>{t('dropImage')}</span>}
                          </label>
                        </div>
                      </div>
                      <div className="form-group">
                        <label>{t('audioLabel')}</label>
                        <div className="file-upload">
                          <input type="file" accept="audio/*" onChange={handleAudioUpload} id="audio-upload" />
                          <label htmlFor="audio-upload" className="upload-area small">
                            {audioFile ? <span>{audioFile.name}</span> : <span>{t('dropAudio')}</span>}
                          </label>
                        </div>
                      </div>
                      <div className="file-pickers">
                        <button className="btn secondary small" onClick={toggleImagePicker}>{t('selectFromUploads')} ({t('imageLabel')})</button>
                        <button className="btn secondary small" onClick={toggleAudioPicker}>{t('selectFromUploads')} ({t('audioLabel')})</button>
                        <button className="btn secondary small" onClick={toggleOutputPicker} disabled={isExtractingFrame}>
                          {isExtractingFrame ? '...' : t('selectFromOutputs')}
                        </button>
                      </div>
                      {showImagePicker && (
                        <div className="picker-list">
                          {uploadedImages.length === 0 ? <p className="picker-empty">{t('noUploads')}</p> : uploadedImages.map((img) => (
                            <div key={img.filename} className={`picker-item${imagePath === img.path ? ' selected' : ''}`} onClick={() => selectUploadedImage(img)}>
                              <img src={img.url} alt={img.filename} className="picker-thumb" />
                              <div className="picker-info">
                                <span className="picker-name">{img.filename}</span>
                                <span className="picker-meta">{img.width}x{img.height} / {(img.size / 1024 / 1024).toFixed(1)}MB</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      {showAudioPicker && (
                        <div className="picker-list">
                          {uploadedAudioList.length === 0 ? <p className="picker-empty">{t('noUploads')}</p> : uploadedAudioList.map((audio) => (
                            <div key={audio.filename} className={`picker-item${audioPath === audio.path ? ' selected' : ''}`} onClick={() => selectUploadedAudio(audio)}>
                              <div className="picker-info">
                                <span className="picker-name">{audio.filename}</span>
                                <span className="picker-meta">{(audio.size / 1024 / 1024).toFixed(1)}MB</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                      {showOutputPicker && (
                        <div className="picker-list">
                          {generatedOutputs.length === 0 ? (
                            <p className="picker-empty">{t('noOutputs')}</p>
                          ) : (
                            generatedOutputs.map((output) => (
                              <div key={output.filename} className="picker-item" onClick={() => selectOutputAsReference(output)}>
                                {output.type === 'image' ? (
                                  <img src={output.url} alt={output.filename} className="picker-thumb" />
                                ) : (
                                  <video src={output.url} className="picker-thumb" preload="metadata" />
                                )}
                                <div className="picker-info">
                                  <span className="picker-name">{output.filename}</span>
                                  <span className="picker-meta">
                                    {output.type === 'image' ? t('outputTypeImage') : t('outputTypeVideo')}
                                    {' / '}
                                    {(output.size / 1024 / 1024).toFixed(1)}MB
                                  </span>
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      )}
                    </div>

                    <div className="card">
                      <h3>Settings</h3>
                      <div className="form-group"><label>{t('promptLabel')}</label><textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={3} /></div>
                      <div className="form-group"><label>{t('negPromptLabel')}</label><textarea value={negPrompt} onChange={(e) => setNegPrompt(e.target.value)} rows={2} /></div>
                      <div className="form-group checkbox">
                        <label><input type="checkbox" checked={autoResolution} onChange={(e) => { setAutoResolution(e.target.checked); if (e.target.checked && imageDimensions) setResolution(`${imageDimensions.height}*${imageDimensions.width}`); }} /> {t('autoResolution')}</label>
                        {imageDimensions && <span className="image-size-info">{t('imageSize')}: {imageDimensions.width} x {imageDimensions.height}</span>}
                      </div>
                      <div className="form-row">
                        <div className="form-group">
                          <label>{t('resolutionLabel')}: {resolution}</label>
                          {autoResolution ? <input type="text" value={resolution} onChange={(e) => setResolution(e.target.value)} placeholder="height*width" /> : (
                            <select value={resolution} onChange={(e) => setResolution(e.target.value)}>
                              {config?.resolutions?.map((r) => <option key={r} value={r}>{r}</option>) || <><option value="720*1280">720*1280</option><option value="1280*720">1280*720</option></>}
                            </select>
                          )}
                        </div>
                        <div className="form-group"><label>{t('clipsLabel')}</label><input type="number" min={0} max={10} value={numClips} onChange={(e) => setNumClips(parseInt(e.target.value))} /></div>
                      </div>
                      <div className="form-row">
                        <div className="form-group"><label>{t('stepsLabel')}: {steps}</label><input type="range" min={5} max={50} step={1} value={steps} onChange={(e) => setSteps(parseInt(e.target.value))} /></div>
                        <div className="form-group"><label>{t('guidanceLabel')}: {guidance}</label><input type="range" min={1} max={10} step={0.5} value={guidance} onChange={(e) => setGuidance(parseFloat(e.target.value))} /></div>
                      </div>
                      <div className="form-row">
                        <div className="form-group"><label>{t('framesLabel')}: {frames}</label><input type="range" min={48} max={120} step={4} value={frames} onChange={(e) => setFrames(parseInt(e.target.value))} /></div>
                        <div className="form-group"><label>{t('seedLabel')}</label><input type="number" value={seed} onChange={(e) => setSeed(parseInt(e.target.value))} /></div>
                      </div>
                      <div className="form-group checkbox"><label><input type="checkbox" checked={offload} onChange={(e) => setOffload(e.target.checked)} /> {t('offloadLabel')}</label></div>
                      <div className="form-group checkbox"><label><input type="checkbox" checked={useTeacache} onChange={(e) => setUseTeacache(e.target.checked)} /> {t('teacacheLabel')}</label></div>
                      {useTeacache && <div className="form-group"><label>{t('teacacheThreshLabel')}: {teacacheThresh}</label><input type="range" min={0.05} max={1.0} step={0.05} value={teacacheThresh} onChange={(e) => setTeacacheThresh(parseFloat(e.target.value))} /></div>}
                      <button className="btn primary" onClick={handleGenerate} disabled={isGenerating || !imagePath || !audioPath}>{isGenerating ? t('generating') : t('generateBtn')}</button>
                    </div>
                  </div>

                  <div className="column">
                    <div className="card">
                      <h3>Output</h3>
                      {isGenerating && <div className="progress-container"><div className="progress-bar"><div className="progress-fill" style={{ width: `${progress}%` }} /></div><span className="progress-text">{Math.round(progress)}%</span></div>}
                      <div className="video-container">{outputVideo ? <video controls src={outputVideo} /> : <div className="placeholder"><span>{t('videoOutput')}</span></div>}</div>
                      {outputSeed && <div className="form-group"><label>{t('seedOutput')}</label><input type="text" value={outputSeed} readOnly /></div>}
                      <div className="status-box"><label>{t('status')}</label><p>{status || 'Ready'}</p></div>
                    </div>
                  </div>
                </div>
              )}

              {/* Extract Audio */}
              {lipsyncSubTab === 'extract' && (
                <div className="two-column">
                  <div className="column">
                    <div className="card">
                      <h3>{t('videoInput')}</h3>
                      <div className="file-upload">
                        <input type="file" accept="video/*" onChange={handleExtractVideoUpload} id="extract-video-upload" />
                        <label htmlFor="extract-video-upload" className="upload-area">{extractVideoFile ? <span>{extractVideoFile.name}</span> : <span>{t('dropVideo')}</span>}</label>
                      </div>
                      <button className="btn primary" onClick={handleExtract} disabled={isExtracting || !extractVideoPath}>{isExtracting ? '...' : t('extractBtn')}</button>
                    </div>
                  </div>
                  <div className="column">
                    <div className="card">
                      <h3>{t('audioOutput')}</h3>
                      {extractedAudio ? <audio controls src={extractedAudio} /> : <div className="placeholder small"><span>{t('audioOutput')}</span></div>}
                      <div className="status-box"><label>{t('status')}</label><p>{extractStatus || 'Ready'}</p></div>
                    </div>
                  </div>
                </div>
              )}

              {/* Separate Vocals */}
              {lipsyncSubTab === 'separate' && (
                <div className="two-column">
                  <div className="column">
                    <div className="card">
                      <h3>{t('audioInput')}</h3>
                      <div className="file-upload">
                        <input type="file" accept="audio/*" onChange={handleSeparateAudioUpload} id="separate-audio-upload" />
                        <label htmlFor="separate-audio-upload" className="upload-area">{separateAudioFile ? <span>{separateAudioFile.name}</span> : <span>{t('dropAudio')}</span>}</label>
                      </div>
                      <button className="btn primary" onClick={handleSeparate} disabled={isSeparating || !separateAudioPath}>{isSeparating ? '...' : t('separateBtn')}</button>
                    </div>
                  </div>
                  <div className="column">
                    <div className="card">
                      <h3>{t('vocalsOutput')}</h3>
                      {separatedVocals ? <audio controls src={separatedVocals} /> : <div className="placeholder small"><span>{t('vocalsOutput')}</span></div>}
                      <div className="status-box"><label>{t('status')}</label><p>{separateStatus || 'Ready'}</p></div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ============ GALLERY ============ */}
          {activeMenu === 'gallery' && (
            <div className="page-content">
              <div className="card">
                <div className="gallery-header">
                  <h3>{t('galleryTitle')}</h3>
                  <button className="btn secondary" onClick={fetchGallery} disabled={galleryLoading}>{galleryLoading ? '...' : t('galleryRefresh')}</button>
                </div>

                <div className="sub-tabs">
                  <button className={galleryTab === 'images' ? 'active' : ''} onClick={() => setGalleryTab('images')}>
                    {t('galleryImages')} ({galleryImages.length})
                  </button>
                  <button className={galleryTab === 'videos' ? 'active' : ''} onClick={() => setGalleryTab('videos')}>
                    {t('galleryVideos')} ({videos.length})
                  </button>
                </div>

                {galleryTab === 'images' && (
                  galleryImages.length === 0 ? (
                    <div className="gallery-empty"><p>{t('galleryEmpty')}</p></div>
                  ) : (
                    <div className="gallery-grid">
                      {galleryImages.map((img) => (
                        <div key={img.filename} className="gallery-item">
                          <img src={img.url} alt={img.filename} className="gallery-item-img" />
                          <div className="gallery-item-info">
                            <span className="gallery-item-name" title={img.filename}>{img.filename}</span>
                            <span className="gallery-item-meta">{t('gallerySize')}: {(img.size / 1024 / 1024).toFixed(1)} MB</span>
                            <span className="gallery-item-meta">{t('galleryDate')}: {new Date(img.created_at).toLocaleString()}</span>
                            <button className="btn danger small" onClick={() => handleDeleteOutput(img.filename)}>{t('galleryDelete')}</button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )
                )}

                {galleryTab === 'videos' && (
                  videos.length === 0 ? (
                    <div className="gallery-empty"><p>{t('galleryEmpty')}</p></div>
                  ) : (
                    <div className="gallery-grid">
                      {videos.map((video) => (
                        <div key={video.filename} className="gallery-item">
                          <video controls src={video.url} preload="metadata" />
                          <div className="gallery-item-info">
                            <span className="gallery-item-name" title={video.filename}>{video.filename}</span>
                            <span className="gallery-item-meta">{t('gallerySize')}: {(video.size / 1024 / 1024).toFixed(1)} MB</span>
                            <span className="gallery-item-meta">{t('galleryDate')}: {new Date(video.created_at).toLocaleString()}</span>
                            <button className="btn danger small" onClick={() => handleDeleteOutput(video.filename)}>{t('galleryDelete')}</button>
                          </div>
                        </div>
                      ))}
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
