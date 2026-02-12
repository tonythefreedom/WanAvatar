import { useState, useEffect, useRef, useCallback } from 'react';
import {
  uploadImage,
  uploadAudio,
  uploadVideo,
  startI2VGeneration,
  startFluxGeneration,
  getTaskStatus,
  getConfig,
  getLoraAdapters,
  deleteOutput,
  listUploadedImages,
  getT2iStatus,
  listOutputs,
  startWorkflowGeneration,
  getWorkflowStatus,
  downloadYoutube,
  getWorkflows,
  prepareWorkflowImages,
  generateTTS,
  getTTSSpeakers,
  sendStudioChat,
  trimVideo,
  uploadBackground,
  listBackgrounds,
  listAvatarGroups,
  listAvatarImages,
  getFashionStyles,
  cancelGeneration,
  authLogin,
  authGoogle,
  authMe,
  adminListUsers,
  adminApproveUser,
  adminSuspendUser,
  adminActivateUser,
  adminDeleteUser,
} from './api';
import './App.css';

// Translations
const translations = {
  en: {
    title: 'WanAvatar',
    menuVideoGen: 'Video Gen',
    menuGallery: 'Gallery',
    // Shared
    status: 'Status',
    seedOutput: 'Used Seed',
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
    i2vPromptDefault: 'UlzzangG1, a beautiful young korean woman with large eyes, pale skin, makeup and small full lips, uka, A cinematic video with natural motion, high quality, smooth movement',
    i2vNegPromptDefault: 'ugly, blurry, low quality, distorted, deformed, static, frozen',
    i2vFrameNumLabel: 'Frame Count',
    i2vShiftLabel: 'Noise Shift',
    i2vPortrait: 'Portrait (9:16)',
    i2vLandscape: 'Landscape (16:9)',
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
    loraOnePerType: 'Select one',
    loraMultiSelect: 'Multiple allowed',
    loraTriggerWords: 'Trigger Words',
    loraDescription: 'Description',
    loraInfo: 'Info',
    loraNoAdapters: 'No LoRA adapters available',
    loraHighTip: 'Controls structure, motion, camera (early diffusion steps)',
    loraLowTip: 'Controls appearance, face, texture (late diffusion steps)',
    loraCivitai: 'CivitAI Page',
    loraPreview: 'Preview',
    loraCopied: 'Copied!',
    loraAddToPrompt: 'Add to prompt',
    loraCopyToClipboard: 'Copy',
    loraRec: 'Rec',
    loraClickToReset: 'Click to reset to recommended value',
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
    fluxAspectRatio: 'Aspect Ratio',
    fluxPortrait: 'Portrait (9:16)',
    fluxLandscape: 'Landscape (16:9)',
    fluxSquare: 'Square (1:1)',
    fluxUpscaleLabel: 'Upscale x2 (Real-ESRGAN)',
    fluxGenerateBtn: 'Generate Image',
    fluxGenerating: 'Generating...',
    fluxModelNote: 'FLUX.2-klein-9B: 4-step fast generation. First use requires model download.',
    // Gallery
    galleryImages: 'Images',
    galleryVideos: 'Videos',
    // Output picker
    selectFromOutputs: 'Select from generated outputs',
    noOutputs: 'No generated outputs available',
    outputTypeImage: 'Image',
    outputTypeVideo: 'Video (first frame)',
    // Workflow (generic)
    menuWorkflow: 'Workflow',
    wfGenerateBtn: 'Run Workflow',
    wfGenerating: 'Running...',
    wfCancelBtn: 'Cancel',
    wfCancelled: 'Cancelled',
    wfVideoUpload: 'File Upload',
    wfVideoYoutube: 'YouTube URL',
    wfDownloadBtn: 'Download',
    wfDownloading: 'Downloading...',
    wfSelectImages: 'Select Images',
    wfSelectedCount: 'selected',
    wfVideoEdit: 'Video Edit',
    wfTrimStart: 'Start',
    wfTrimEnd: 'End',
    wfTrimApply: 'Apply Trim',
    wfDuration: 'Duration',
    wfReplaceAudio: 'Replace Audio',
    wfTrimming: 'Trimming...',
    wfRemoveAudio: 'Remove',
    wfBgGallery: 'Select from Stages',
    wfBgUpload: 'Upload New',
    wfBgGalleryEmpty: 'No backgrounds saved yet',
    wfAddToQueue: 'Add to Queue',
    wfQueue: 'Queue',
    wfStartQueue: 'Start Queue',
    wfClearQueue: 'Clear Completed',
    wfQueueEmpty: 'Queue is empty',
    wfQueueRunning: 'Processing queue...',
    wfQueuePending: 'Pending',
    ytMeta: 'YouTube Shorts',
    ytTitle: 'Title',
    ytDescription: 'Description',
    ytHashtags: 'Hashtags',
    ytTitlePlaceholder: 'Auto: {avatar} - date',
    ytDescPlaceholder: 'Video description...',
    ytHashPlaceholder: '#Shorts #AI #dance',
    fashionStyle: 'Fashion Style',
    fashionAll: 'All',
    fashionApply: 'Apply',
    fashionRandom: 'Random',
    output: 'Output',
    download: 'Download',
    noOutputYet: 'No output yet',
    dropImageHere: 'Drop image here or click to upload',
    dropVideoHere: 'Drop video here or click to upload',
    // Studio
    menuStudio: 'Video Studio',
    studioManualMode: 'Manual',
    studioAutoMode: 'AI Assistant',
    studioStep1: '1. Generate Images',
    studioStep2: '2. Arrange Timeline',
    studioStep3: '3. Create Video',
    studioGenerateImage: 'Generate Image',
    studioAddToTimeline: 'Add to Timeline',
    studioCreateVideo: 'Create Video',
    studioSegmentFrames: 'Frames',
    studioDragToReorder: 'Drag to reorder',
    studioSelectFromGallery: 'Select from Gallery',
    studioSeed: 'Seed (same = consistent character)',
    studioFflfMode: 'FFLF Video',
    studioInfiniTalkMode: 'InfiniTalk',
    studioChangeCharMode: 'Change Character',
    studioRefVideo: 'Reference Video',
    studioSceneDesc: 'Scene Description',
    studioAspectRatio: 'Aspect Ratio',
    studioContentType: 'Content Type',
    studioContentDance: 'Dance',
    studioContentNarration: 'Narration',
    studioContentPresentation: 'Presentation',
    studioNarrationDesc: 'Narration video from image + audio (unlimited duration)',
    studioPresentationDesc: 'Presentation with smooth transitions',
    studioDanceDesc: 'Dance/choreography with looping motion',
    studioUploadAudio: 'Upload Audio File',
    studioTtsScript: 'Narration Script',
    studioTtsGenerate: 'Generate Voice',
    studioTtsLanguage: 'Language',
    studioTtsSpeaker: 'Speaker',
    studioAudioSourceTts: 'Generate from Script (TTS)',
    studioAudioSourceUpload: 'Upload Audio File',
    studioMasterPrompt: 'Master Prompt',
    studioNegPrompt: 'Negative Prompt',
    studioLooping: 'Looping',
    studioInitialWidth: 'Initial Width',
    studioUpscaleFactor: 'Upscale Factor',
    studioVideoSeed: 'Video Seed',
    studioMinImages: 'At least 2 images required for FFLF',
    studioNeedImage: 'Generate a character image first',
    studioNeedAudio: 'Upload or generate narration audio first',
    studioNeedRefVideo: 'Upload a reference video first',
    studioTotalFrames: 'Total Frames',
    studioEstDuration: 'Est. Duration',
    // Auth
    loginTitle: 'Sign In',
    loginEmail: 'Email',
    loginPassword: 'Password',
    loginSignIn: 'Sign In',
    loginGoogleBtn: 'Sign in with Google',
    loginPending: 'Your account is pending approval. An administrator will review your registration.',
    loginError: 'Login failed',
    rememberEmail: 'Remember ID',
    logoutBtn: 'Logout',
    // Admin
    menuAdmin: 'Users',
    adminTitle: 'User Management',
    adminEmail: 'Email',
    adminName: 'Name',
    adminRole: 'Role',
    adminStatus: 'Status',
    adminCreated: 'Joined',
    adminLastLogin: 'Last Login',
    adminActions: 'Actions',
    adminApprove: 'Approve',
    adminSuspend: 'Suspend',
    adminActivate: 'Activate',
    adminDelete: 'Delete',
    adminDeleteConfirm: 'Delete this user?',
    adminPending: 'pending',
    adminApproved: 'approved',
    adminSuspended: 'suspended',
  },
  ko: {
    title: 'WanAvatar',
    menuVideoGen: '비디오 생성',
    menuGallery: '갤러리',
    status: '상태',
    seedOutput: '사용된 시드',
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
    i2vPromptDefault: 'UlzzangG1, a beautiful young korean woman with large eyes, pale skin, makeup and small full lips, uka, A cinematic video with natural motion, high quality, smooth movement',
    i2vNegPromptDefault: 'ugly, blurry, low quality, distorted, deformed, static, frozen',
    i2vFrameNumLabel: '프레임 수',
    i2vShiftLabel: '노이즈 시프트',
    i2vPortrait: '세로 (9:16)',
    i2vLandscape: '가로 (16:9)',
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
    loraOnePerType: '하나만 선택',
    loraMultiSelect: '다중 선택 가능',
    loraTriggerWords: '트리거 워드',
    loraDescription: '설명',
    loraInfo: '정보',
    loraNoAdapters: '사용 가능한 LoRA 어댑터 없음',
    loraHighTip: '구조, 동작, 카메라 제어 (초기 확산 단계)',
    loraLowTip: '외관, 얼굴, 텍스처 제어 (후기 확산 단계)',
    loraCivitai: 'CivitAI 페이지',
    loraPreview: '미리보기',
    loraCopied: '복사됨!',
    loraAddToPrompt: '프롬프트에 추가',
    loraCopyToClipboard: '복사',
    loraRec: '추천',
    loraClickToReset: '클릭하면 추천값으로 리셋',
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
    fluxAspectRatio: '비율',
    fluxPortrait: '세로 (9:16)',
    fluxLandscape: '가로 (16:9)',
    fluxSquare: '정사각 (1:1)',
    fluxUpscaleLabel: '업스케일 x2 (Real-ESRGAN)',
    fluxGenerateBtn: '이미지 생성',
    fluxGenerating: '생성 중...',
    fluxModelNote: 'FLUX.2-klein-9B: 4스텝 고속 생성. 첫 사용 시 모델 다운로드 필요.',
    // Gallery
    galleryImages: '이미지',
    galleryVideos: '동영상',
    // Output picker
    selectFromOutputs: '생성된 결과에서 선택',
    noOutputs: '생성된 결과가 없습니다',
    outputTypeImage: '이미지',
    outputTypeVideo: '비디오 (첫 프레임)',
    // Workflow (generic)
    menuWorkflow: '워크플로우',
    wfGenerateBtn: '워크플로우 실행',
    wfGenerating: '실행 중...',
    wfCancelBtn: '취소',
    wfCancelled: '취소됨',
    wfVideoUpload: '파일 업로드',
    wfVideoYoutube: 'YouTube URL',
    wfDownloadBtn: '다운로드',
    wfDownloading: '다운로드 중...',
    wfSelectImages: '이미지 선택',
    wfSelectedCount: '선택됨',
    wfVideoEdit: '비디오 편집',
    wfTrimStart: '시작',
    wfTrimEnd: '종료',
    wfTrimApply: '트림 적용',
    wfDuration: '길이',
    wfReplaceAudio: '오디오 교체',
    wfTrimming: '트리밍 중...',
    wfRemoveAudio: '제거',
    wfBgGallery: '스테이지에서 선택',
    wfBgUpload: '새로 업로드',
    wfBgGalleryEmpty: '저장된 배경이 없습니다',
    wfAddToQueue: '큐에 추가',
    wfQueue: '큐',
    wfStartQueue: '큐 시작',
    wfClearQueue: '완료 항목 정리',
    wfQueueEmpty: '큐가 비어있습니다',
    wfQueueRunning: '큐 처리 중...',
    wfQueuePending: '대기 중',
    ytMeta: 'YouTube Shorts',
    ytTitle: '제목',
    ytDescription: '설명',
    ytHashtags: '해시태그',
    ytTitlePlaceholder: '자동: {아바타명} - 날짜',
    ytDescPlaceholder: '영상 설명...',
    ytHashPlaceholder: '#Shorts #AI #dance',
    fashionStyle: '패션 스타일',
    fashionAll: '전체',
    fashionApply: '적용',
    fashionRandom: '랜덤',
    output: '출력',
    download: '다운로드',
    noOutputYet: '아직 출력이 없습니다',
    dropImageHere: '이미지를 드롭하거나 클릭하여 업로드',
    dropVideoHere: '비디오를 드롭하거나 클릭하여 업로드',
    // Studio
    menuStudio: '비디오 스튜디오',
    studioManualMode: '수동',
    studioAutoMode: 'AI 어시스턴트',
    studioStep1: '1. 이미지 생성',
    studioStep2: '2. 타임라인 배치',
    studioStep3: '3. 비디오 생성',
    studioGenerateImage: '이미지 생성',
    studioAddToTimeline: '타임라인에 추가',
    studioCreateVideo: '비디오 생성',
    studioSegmentFrames: '프레임',
    studioDragToReorder: '드래그하여 순서 변경',
    studioSelectFromGallery: '갤러리에서 선택',
    studioSeed: '시드 (동일 = 일관된 캐릭터)',
    studioFflfMode: 'FFLF 비디오',
    studioInfiniTalkMode: '인피니톡',
    studioChangeCharMode: '캐릭터 교체',
    studioRefVideo: '참조 비디오',
    studioSceneDesc: '장면 설명',
    studioAspectRatio: '비율',
    studioContentType: '콘텐츠 유형',
    studioContentDance: '댄스',
    studioContentNarration: '나레이션',
    studioContentPresentation: '프리젠테이션',
    studioNarrationDesc: '이미지+오디오 나레이션 영상 (무제한 길이)',
    studioPresentationDesc: '부드러운 전환의 프리젠테이션 영상',
    studioDanceDesc: '루핑 모션의 댄스/안무 영상',
    studioUploadAudio: '오디오 파일 업로드',
    studioTtsScript: '나레이션 대본',
    studioTtsGenerate: '음성 생성',
    studioTtsLanguage: '언어',
    studioTtsSpeaker: '화자',
    studioAudioSourceTts: '대본에서 생성 (TTS)',
    studioAudioSourceUpload: '오디오 파일 업로드',
    studioMasterPrompt: '마스터 프롬프트',
    studioNegPrompt: '네거티브 프롬프트',
    studioLooping: '루핑',
    studioInitialWidth: '초기 너비',
    studioUpscaleFactor: '업스케일 배율',
    studioVideoSeed: '비디오 시드',
    studioMinImages: 'FFLF에는 최소 2장의 이미지가 필요합니다',
    studioNeedImage: '먼저 캐릭터 이미지를 생성하세요',
    studioNeedAudio: '먼저 나레이션 오디오를 업로드하거나 생성하세요',
    studioNeedRefVideo: '참조 비디오를 업로드하세요',
    studioTotalFrames: '총 프레임',
    studioEstDuration: '예상 시간',
    // Auth
    loginTitle: '로그인',
    loginEmail: '이메일',
    loginPassword: '비밀번호',
    loginSignIn: '로그인',
    loginGoogleBtn: 'Google로 로그인',
    loginPending: '계정 승인 대기 중입니다. 관리자가 가입을 검토할 예정입니다.',
    loginError: '로그인 실패',
    rememberEmail: 'ID 저장',
    logoutBtn: '로그아웃',
    // Admin
    menuAdmin: '사용자 관리',
    adminTitle: '사용자 관리',
    adminEmail: '이메일',
    adminName: '이름',
    adminRole: '역할',
    adminStatus: '상태',
    adminCreated: '가입일',
    adminLastLogin: '마지막 로그인',
    adminActions: '관리',
    adminApprove: '승인',
    adminSuspend: '정지',
    adminActivate: '활성화',
    adminDelete: '삭제',
    adminDeleteConfirm: '이 사용자를 삭제하시겠습니까?',
    adminPending: '대기',
    adminApproved: '승인됨',
    adminSuspended: '정지됨',
  },
  zh: {
    title: 'WanAvatar',
    menuVideoGen: '视频生成',
    menuGallery: '画廊',
    status: '状态',
    seedOutput: '使用的种子',
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
    i2vPromptDefault: 'UlzzangG1, a beautiful young korean woman with large eyes, pale skin, makeup and small full lips, uka, A cinematic video with natural motion, high quality, smooth movement',
    i2vNegPromptDefault: 'ugly, blurry, low quality, distorted, deformed, static, frozen',
    i2vFrameNumLabel: '帧数',
    i2vShiftLabel: '噪声偏移',
    i2vPortrait: '竖屏 (9:16)',
    i2vLandscape: '横屏 (16:9)',
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
    loraOnePerType: '选择一个',
    loraMultiSelect: '可多选',
    loraTriggerWords: '触发词',
    loraDescription: '描述',
    loraInfo: '信息',
    loraNoAdapters: '无可用LoRA适配器',
    loraHighTip: '控制结构、运动、镜头（扩散早期步骤）',
    loraLowTip: '控制外观、面部、纹理（扩散后期步骤）',
    loraCivitai: 'CivitAI页面',
    loraPreview: '预览',
    loraCopied: '已复制!',
    loraAddToPrompt: '添加到提示词',
    loraCopyToClipboard: '复制',
    loraRec: '推荐',
    loraClickToReset: '点击重置为推荐值',
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
    fluxAspectRatio: '比例',
    fluxPortrait: '竖屏 (9:16)',
    fluxLandscape: '横屏 (16:9)',
    fluxSquare: '正方形 (1:1)',
    fluxUpscaleLabel: '放大 x2 (Real-ESRGAN)',
    fluxGenerateBtn: '生成图像',
    fluxGenerating: '生成中...',
    fluxModelNote: 'FLUX.2-klein-9B: 4步快速生成。首次使用需下载模型。',
    // Gallery
    galleryImages: '图片',
    galleryVideos: '视频',
    // Output picker
    selectFromOutputs: '从生成结果中选择',
    noOutputs: '没有生成结果',
    outputTypeImage: '图片',
    outputTypeVideo: '视频（首帧）',
    // Workflow
    menuWorkflow: '工作流',
    wfGenerateBtn: '运行工作流',
    wfGenerating: '运行中...',
    wfCancelBtn: '取消',
    wfCancelled: '已取消',
    wfVideoUpload: '文件上传',
    wfVideoYoutube: 'YouTube链接',
    wfDownloadBtn: '下载',
    wfDownloading: '下载中...',
    wfSelectImages: '选择图片',
    wfSelectedCount: '已选择',
    wfVideoEdit: '视频编辑',
    wfTrimStart: '开始',
    wfTrimEnd: '结束',
    wfTrimApply: '应用裁剪',
    wfDuration: '时长',
    wfReplaceAudio: '替换音频',
    wfTrimming: '裁剪中...',
    wfRemoveAudio: '删除',
    wfBgGallery: '从舞台选择',
    wfBgUpload: '上传新背景',
    wfBgGalleryEmpty: '暂无保存的背景',
    wfAddToQueue: '添加到队列',
    wfQueue: '队列',
    wfStartQueue: '开始队列',
    wfClearQueue: '清除已完成',
    wfQueueEmpty: '队列为空',
    wfQueueRunning: '队列处理中...',
    wfQueuePending: '等待中',
    ytMeta: 'YouTube Shorts',
    ytTitle: '标题',
    ytDescription: '描述',
    ytHashtags: '标签',
    ytTitlePlaceholder: '自动: {角色名} - 日期',
    ytDescPlaceholder: '视频描述...',
    ytHashPlaceholder: '#Shorts #AI #dance',
    fashionStyle: '时尚风格',
    fashionAll: '全部',
    fashionApply: '应用',
    fashionRandom: '随机',
    output: '输出',
    download: '下载',
    noOutputYet: '暂无输出',
    dropImageHere: '将图片拖放到此处或点击上传',
    dropVideoHere: '将视频拖放到此处或点击上传',
    // Studio
    menuStudio: '视频工作室',
    studioManualMode: '手动',
    studioAutoMode: 'AI助手',
    studioStep1: '1. 生成图片',
    studioStep2: '2. 排列时间线',
    studioStep3: '3. 生成视频',
    studioGenerateImage: '生成图片',
    studioAddToTimeline: '添加到时间线',
    studioCreateVideo: '生成视频',
    studioSegmentFrames: '帧数',
    studioDragToReorder: '拖拽重新排序',
    studioSelectFromGallery: '从画廊选择',
    studioSeed: '种子 (相同=角色一致)',
    studioFflfMode: 'FFLF视频',
    studioInfiniTalkMode: 'InfiniTalk',
    studioChangeCharMode: '角色替换',
    studioRefVideo: '参考视频',
    studioSceneDesc: '场景描述',
    studioAspectRatio: '比例',
    studioContentType: '内容类型',
    studioContentDance: '舞蹈',
    studioContentNarration: '旁白',
    studioContentPresentation: '演示',
    studioNarrationDesc: '图片+音频旁白视频(无限时长)',
    studioPresentationDesc: '平滑过渡的演示视频',
    studioDanceDesc: '循环动作的舞蹈视频',
    studioUploadAudio: '上传音频文件',
    studioTtsScript: '旁白脚本',
    studioTtsGenerate: '生成语音',
    studioTtsLanguage: '语言',
    studioTtsSpeaker: '说话人',
    studioAudioSourceTts: '从脚本生成 (TTS)',
    studioAudioSourceUpload: '上传音频文件',
    studioMasterPrompt: '主提示词',
    studioNegPrompt: '负面提示词',
    studioLooping: '循环',
    studioInitialWidth: '初始宽度',
    studioUpscaleFactor: '放大倍数',
    studioVideoSeed: '视频种子',
    studioMinImages: 'FFLF至少需要2张图片',
    studioNeedImage: '请先生成角色图片',
    studioNeedAudio: '请先上传或生成旁白音频',
    studioNeedRefVideo: '请上传参考视频',
    studioTotalFrames: '总帧数',
    studioEstDuration: '预计时长',
    // Auth
    loginTitle: '登录',
    loginEmail: '邮箱',
    loginPassword: '密码',
    loginSignIn: '登录',
    loginGoogleBtn: '使用Google登录',
    loginPending: '您的账户正在等待审批。管理员将审核您的注册。',
    loginError: '登录失败',
    rememberEmail: '记住ID',
    logoutBtn: '退出',
    // Admin
    menuAdmin: '用户管理',
    adminTitle: '用户管理',
    adminEmail: '邮箱',
    adminName: '姓名',
    adminRole: '角色',
    adminStatus: '状态',
    adminCreated: '注册时间',
    adminLastLogin: '最近登录',
    adminActions: '操作',
    adminApprove: '批准',
    adminSuspend: '暂停',
    adminActivate: '激活',
    adminDelete: '删除',
    adminDeleteConfirm: '确定删除此用户？',
    adminPending: '待审批',
    adminApproved: '已批准',
    adminSuspended: '已暂停',
  },
};

function App() {
  const [lang, setLang] = useState('en');
  const [activeMenu, setActiveMenu] = useState('studio');
  const [config, setConfig] = useState(null);

  // Auth state
  const [authUser, setAuthUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [isPendingApproval, setIsPendingApproval] = useState(false);
  const [loginEmail, setLoginEmail] = useState(() => localStorage.getItem('saved_login_email') || '');
  const [loginPassword, setLoginPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [loginLoading, setLoginLoading] = useState(false);
  const [rememberEmail, setRememberEmail] = useState(() => !!localStorage.getItem('saved_login_email'));
  // Admin state
  const [adminUsers, setAdminUsers] = useState([]);
  const [adminLoading, setAdminLoading] = useState(false);

  // Gallery state
  const [videos, setVideos] = useState([]);
  const [galleryLoading, setGalleryLoading] = useState(false);

  // I2V state
  const [i2vImageFile, setI2vImageFile] = useState(null);
  const [i2vImagePreview, setI2vImagePreview] = useState(null);
  const [i2vImagePath, setI2vImagePath] = useState('');
  const [i2vImageDimensions, setI2vImageDimensions] = useState(null);
  const [i2vPrompt, setI2vPrompt] = useState(translations.en.i2vPromptDefault);
  const [i2vNegPrompt, setI2vNegPrompt] = useState(translations.en.i2vNegPromptDefault);
  const [i2vResolution, setI2vResolution] = useState('1280*720');
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
  const [showI2vOutputPicker, setShowI2vOutputPicker] = useState(false);
  const [i2vGeneratedImages, setI2vGeneratedImages] = useState([]);

  // LoRA state
  const [loraAdapters, setLoraAdapters] = useState([]);
  const [loraWeights, setLoraWeights] = useState({}); // {name: {enabled, high_weight, low_weight}}
  const [expandedLora, setExpandedLora] = useState(null); // name of expanded info panel

  // Shared upload state (used by Workflow page)
  const [uploadedImages, setUploadedImages] = useState([]);

  // Image Gen state
  const [imgGenImages, setImgGenImages] = useState([]);
  const [t2iAvailable, setT2iAvailable] = useState(false);
  const [t2iMessage, setT2iMessage] = useState('');

  // FLUX generation state
  const [fluxPrompt, setFluxPrompt] = useState('K-pop idol, young Korean female, symmetrical face, V-shaped jawline, clear glass skin, double eyelids, trendy idol makeup.\n\nStage lighting, cinematic bokeh, pink and purple neon highlights, professional studio portrait, high-end fashion editorial style.\n\n8k resolution, photorealistic, raw photo, masterwork, intricate details of eyes and hair.');
  const [fluxSteps, setFluxSteps] = useState(4);
  const [fluxGuidance, setFluxGuidance] = useState(1.0);
  const [fluxSeed, setFluxSeed] = useState(-1);
  const [fluxAspectRatio, setFluxAspectRatio] = useState('portrait');
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

  // Workflow state (dynamic, per-workflow)
  const [workflows, setWorkflows] = useState([]);
  const [activeWorkflowId, setActiveWorkflowId] = useState(null);
  const [workflowStates, setWorkflowStates] = useState({});
  const [wfQueue, setWfQueue] = useState({});
  const wfQueueRef = useRef({});
  const videoRefsMap = useRef({});
  const timelineRefsMap = useRef({});

  // ============ Studio state ============
  const [studioMode, setStudioMode] = useState('manual'); // 'manual' | 'auto'
  const [studioStep, setStudioStep] = useState(1);
  const [studioContentType, setStudioContentType] = useState('dance');

  const CONTENT_PRESETS = {
    dance: { defaultSegmentLength: 33, looping: true, aspectRatio: 'portrait', recommendedMode: 'fflf',
      masterPrompt: 'A beautiful idol dancing smoothly, dynamic lighting, studio background, high quality',
      negPrompt: 'blurry, distorted, low quality, static' },
    narration: { defaultSegmentLength: 65, looping: false, aspectRatio: 'landscape', recommendedMode: 'infinitalk',
      masterPrompt: 'A person speaking to camera, clear expression, natural gestures, professional lighting, high quality',
      negPrompt: 'blurry, distorted, low quality, unnatural motion' },
    presentation: { defaultSegmentLength: 49, looping: false, aspectRatio: 'landscape', recommendedMode: 'fflf',
      masterPrompt: 'Professional product showcase, smooth camera movement, clean background, high quality commercial',
      negPrompt: 'blurry, distorted, low quality, cluttered background' },
  };

  // Step 1
  const [studioFluxPrompt, setStudioFluxPrompt] = useState('');
  const [studioFluxSeed, setStudioFluxSeed] = useState(-1);
  const [studioFluxAspectRatio, setStudioFluxAspectRatio] = useState('portrait');
  const [studioFluxIsGenerating, setStudioFluxIsGenerating] = useState(false);
  const [studioFluxProgress, setStudioFluxProgress] = useState(0);
  const [studioSelectedLoras, setStudioSelectedLoras] = useState({});

  // Step 2
  const [studioTimeline, setStudioTimeline] = useState([]);
  const [studioDragIndex, setStudioDragIndex] = useState(null);

  // Step 3
  const [studioVideoMode, setStudioVideoMode] = useState('fflf');
  const [studioMasterPrompt, setStudioMasterPrompt] = useState(CONTENT_PRESETS.dance.masterPrompt);
  const [studioNegPrompt, setStudioNegPrompt] = useState(CONTENT_PRESETS.dance.negPrompt);
  const [studioIsGenerating, setStudioIsGenerating] = useState(false);
  const [studioProgress, setStudioProgress] = useState(0);
  const [studioStatus, setStudioStatus] = useState('');
  const [studioOutputVideo, setStudioOutputVideo] = useState(null);
  const [studioLooping, setStudioLooping] = useState(true);
  const [studioInitialWidth, setStudioInitialWidth] = useState(288);
  const [studioUpscaleFactor, setStudioUpscaleFactor] = useState(2.5);
  const [studioVideoSeed, setStudioVideoSeed] = useState(138);

  // InfiniTalk
  const [studioNarrationAudio, setStudioNarrationAudio] = useState(null);
  const [studioNarrationAudioPath, setStudioNarrationAudioPath] = useState('');
  const [studioNarrationLength, setStudioNarrationLength] = useState(160);

  // TTS
  const [studioTtsScript, setStudioTtsScript] = useState('');
  const [studioTtsLanguage, setStudioTtsLanguage] = useState('Korean');
  const [studioTtsSpeaker, setStudioTtsSpeaker] = useState('Ryan');
  const [studioTtsGenerating, setStudioTtsGenerating] = useState(false);
  const [studioAudioSource, setStudioAudioSource] = useState('tts');

  // Change Character
  const [studioRefVideoPath, setStudioRefVideoPath] = useState('');
  const [studioRefVideoPreview, setStudioRefVideoPreview] = useState(null);
  const [studioRefVideoMode, setStudioRefVideoMode] = useState('upload');
  const [studioYoutubeUrl, setStudioYoutubeUrl] = useState('');
  const [studioYoutubeDownloading, setStudioYoutubeDownloading] = useState(false);
  const [studioScenePrompt, setStudioScenePrompt] = useState('The character is dancing in the room');

  // Gallery picker
  const [studioGalleryOpen, setStudioGalleryOpen] = useState(false);
  const [studioGalleryImages, setStudioGalleryImages] = useState([]);

  // AI Chat
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);

  const t = useCallback((key) => translations[lang][key] || key, [lang]);

  // ─── Auth: check session on mount ───
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (!token) { setAuthLoading(false); return; }
    authMe()
      .then(data => {
        setAuthUser(data);
        setIsPendingApproval(false);
        setAuthLoading(false);
      })
      .catch(err => {
        if (err.response?.status === 403 && err.response?.data?.detail?.includes('pending')) {
          setIsPendingApproval(true);
        } else {
          localStorage.removeItem('auth_token');
        }
        setAuthLoading(false);
      });
    const onLogout = () => { setAuthUser(null); setIsPendingApproval(false); };
    window.addEventListener('auth-logout', onLogout);
    return () => window.removeEventListener('auth-logout', onLogout);
  }, []);

  // ─── Auth handlers ───
  const handleEmailLogin = async (e) => {
    e.preventDefault();
    setLoginError('');
    setLoginLoading(true);
    // Save or clear remembered email
    if (rememberEmail) {
      localStorage.setItem('saved_login_email', loginEmail);
    } else {
      localStorage.removeItem('saved_login_email');
    }
    try {
      const data = await authLogin(loginEmail, loginPassword);
      localStorage.setItem('auth_token', data.token);
      const me = await authMe();
      setAuthUser(me);
      setIsPendingApproval(false);
    } catch (err) {
      setLoginError(err.response?.data?.detail || t('loginError'));
    } finally {
      setLoginLoading(false);
    }
  };

  const handleGoogleLogin = useCallback(() => {
    if (!window.google?.accounts?.id) return;
    window.google.accounts.id.initialize({
      client_id: config?.google_client_id || '',
      callback: async (response) => {
        setLoginError('');
        setLoginLoading(true);
        try {
          const data = await authGoogle(response.credential);
          localStorage.setItem('auth_token', data.token);
          if (data.status === 'pending') {
            setIsPendingApproval(true);
          } else {
            const me = await authMe();
            setAuthUser(me);
            setIsPendingApproval(false);
          }
        } catch (err) {
          if (err.response?.status === 403 && err.response?.data?.detail?.includes('pending')) {
            setIsPendingApproval(true);
          } else {
            setLoginError(err.response?.data?.detail || t('loginError'));
          }
        } finally {
          setLoginLoading(false);
        }
      },
    });
    window.google.accounts.id.prompt();
  }, [config, t]);

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    setAuthUser(null);
    setIsPendingApproval(false);
  };

  // ─── Admin handlers ───
  const fetchAdminUsers = useCallback(async () => {
    setAdminLoading(true);
    try {
      const data = await adminListUsers();
      setAdminUsers(data.users || []);
    } catch {} finally { setAdminLoading(false); }
  }, []);

  const handleAdminAction = async (userId, action) => {
    try {
      if (action === 'approve') await adminApproveUser(userId);
      else if (action === 'suspend') await adminSuspendUser(userId);
      else if (action === 'activate') await adminActivateUser(userId);
      else if (action === 'delete') {
        if (!confirm(t('adminDeleteConfirm'))) return;
        await adminDeleteUser(userId);
      }
      fetchAdminUsers();
    } catch (err) {
      alert(err.response?.data?.detail || 'Error');
    }
  };

  useEffect(() => {
    if (activeMenu === 'admin' && authUser?.role === 'superadmin') fetchAdminUsers();
  }, [activeMenu, authUser, fetchAdminUsers]);

  // Load config & LoRA adapters
  useEffect(() => {
    getConfig().then(res => setConfig(res.data)).catch(console.error);

    // Load mov-category LoRAs (for Video Gen page)
    getLoraAdapters('mov').then(data => {
      setLoraAdapters(data.adapters || []);
      // Build defaults: preferred adapters enabled, camera off by default
      const defaults = {};
      const adapters = data.adapters || [];
      const defaultEnabled = new Set(['UlzzangG1', 'UkaSexyLight']);
      adapters.forEach(a => {
        defaults[a.name] = {
          enabled: a.available && defaultEnabled.has(a.name),
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

    // Load available workflows
    getWorkflows().then(data => {
      const wfs = data.workflows || [];
      setWorkflows(wfs);
      if (wfs.length > 0) setActiveWorkflowId(wfs[0].id);
      const initStates = {};
      wfs.forEach(wf => {
        initStates[wf.id] = {
          inputs: {}, filePaths: {}, previews: {},
          isGenerating: false, progress: 0, status: '', outputVideo: null, currentTaskId: null,
          videoInputMode: {}, youtubeUrl: {}, youtubeDownloading: {},
          galleryOpen: {}, gallerySelected: {},
          videoDuration: {}, trimStart: {}, trimEnd: {}, trimming: {}, playheadPosition: {},
          bgGalleryOpen: {}, bgGalleryImages: [],
          avatarGroups: [], avatarSelectedGroup: {}, avatarImages: {},
          ytTitle: '', ytDescription: '', ytHashtags: '',
          fashionStyles: [], fashionCategories: [], fashionFilterCat: 'All',
        };
        wf.inputs.forEach(inp => {
          if (inp.default !== undefined) initStates[wf.id].inputs[inp.key] = inp.default;
          if (inp.default_path) {
            initStates[wf.id].filePaths[inp.key] = inp.default_path;
            if (inp.default_preview) initStates[wf.id].previews[inp.key] = inp.default_preview;
          }
          if (inp.type === 'video') {
            initStates[wf.id].videoInputMode[inp.key] = 'upload';
            initStates[wf.id].youtubeUrl[inp.key] = '';
            initStates[wf.id].youtubeDownloading[inp.key] = false;
          }
          if (inp.type === 'gallery_select') {
            initStates[wf.id].galleryOpen[inp.key] = false;
            initStates[wf.id].gallerySelected[inp.key] = [];
          }
        });
      });
      setWorkflowStates(initStates);
    }).catch(console.error);
  }, []);

  // Workflow state helpers
  const updateWfState = useCallback((wfId, updates) => {
    setWorkflowStates(prev => ({ ...prev, [wfId]: { ...prev[wfId], ...updates } }));
  }, []);
  const updateWfInput = useCallback((wfId, key, value) => {
    setWorkflowStates(prev => ({
      ...prev, [wfId]: { ...prev[wfId], inputs: { ...prev[wfId].inputs, [key]: value } },
    }));
  }, []);
  const updateWfFilePath = useCallback((wfId, key, path) => {
    setWorkflowStates(prev => ({
      ...prev, [wfId]: { ...prev[wfId], filePaths: { ...prev[wfId].filePaths, [key]: path } },
    }));
  }, []);
  const updateWfPreview = useCallback((wfId, key, url) => {
    setWorkflowStates(prev => ({
      ...prev, [wfId]: { ...prev[wfId], previews: { ...prev[wfId].previews, [key]: url } },
    }));
  }, []);

  // ============ Studio Handlers ============
  const studioHandleContentTypeChange = (type) => {
    setStudioContentType(type);
    const preset = CONTENT_PRESETS[type];
    setStudioLooping(preset.looping);
    setStudioFluxAspectRatio(preset.aspectRatio);
    setStudioMasterPrompt(preset.masterPrompt);
    setStudioNegPrompt(preset.negPrompt);
    setStudioVideoMode(preset.recommendedMode);
    setStudioTimeline(prev => prev.map(item => ({ ...item, segmentLength: preset.defaultSegmentLength })));
  };

  const studioAddToTimeline = useCallback((imageUrl, imagePath) => {
    const preset = CONTENT_PRESETS[studioContentType];
    setStudioTimeline(prev => [...prev, {
      id: crypto.randomUUID(), imageUrl, imagePath,
      segmentLength: preset.defaultSegmentLength,
    }]);
  }, [studioContentType]);

  const studioRemoveFromTimeline = (index) => {
    setStudioTimeline(prev => prev.filter((_, i) => i !== index));
  };

  const studioUpdateSegmentLength = (index, value) => {
    setStudioTimeline(prev => prev.map((item, i) =>
      i === index ? { ...item, segmentLength: Math.max(17, Math.min(81, value)) } : item
    ));
  };

  const studioHandleDragStart = (e, index) => {
    setStudioDragIndex(index);
    e.dataTransfer.effectAllowed = 'move';
  };
  const studioHandleDragOver = (e, index) => {
    e.preventDefault();
    if (studioDragIndex === null || studioDragIndex === index) return;
    setStudioTimeline(prev => {
      const items = [...prev];
      const [dragged] = items.splice(studioDragIndex, 1);
      items.splice(index, 0, dragged);
      return items;
    });
    setStudioDragIndex(index);
  };
  const studioHandleDragEnd = () => setStudioDragIndex(null);

  // Studio FLUX generate (reuses existing FLUX API)
  const studioHandleFluxGenerate = async () => {
    if (!studioFluxPrompt.trim()) return;
    setStudioFluxIsGenerating(true);
    setStudioFluxProgress(0);
    try {
      const enabledLoras = Object.entries(studioSelectedLoras)
        .filter(([, v]) => v.enabled)
        .map(([name, v]) => ({ name, high_weight: v.high_weight, low_weight: v.low_weight }));
      const { task_id } = await startFluxGeneration({
        prompt: studioFluxPrompt,
        seed: studioFluxSeed,
        aspect_ratio: studioFluxAspectRatio,
        num_inference_steps: 4,
        guidance_scale: 1.0,
        upscale: false,
        lora_weights: enabledLoras.length > 0 ? enabledLoras : undefined,
      });
      const poll = setInterval(async () => {
        try {
          const s = await getTaskStatus(task_id);
          setStudioFluxProgress(Math.round((s.progress || 0) * 100));
          if (s.status === 'completed') {
            clearInterval(poll);
            setStudioFluxIsGenerating(false);
            if (s.output_path) {
              studioAddToTimeline(s.output_path, s.absolute_path || s.output_path);
            }
          } else if (s.status === 'failed') {
            clearInterval(poll);
            setStudioFluxIsGenerating(false);
            alert(`Failed: ${s.message}`);
          }
        } catch { clearInterval(poll); setStudioFluxIsGenerating(false); }
      }, 1500);
    } catch (err) {
      setStudioFluxIsGenerating(false);
      alert(`Error: ${err.message}`);
    }
  };

  // Load gallery images for studio picker
  const studioLoadGallery = async () => {
    try {
      const data = await listOutputs();
      setStudioGalleryImages((data.outputs || []).filter(o => o.type === 'image'));
    } catch { /* ignore */ }
  };

  // Studio FFLF video generation
  const studioHandleCreateVideo = async () => {
    if (studioTimeline.length < 2) return alert(t('studioMinImages'));
    setStudioIsGenerating(true);
    setStudioProgress(0);
    setStudioStatus('Preparing images...');
    try {
      const imagePaths = studioTimeline.map(item => item.imagePath);
      const { folder_path } = await prepareWorkflowImages(imagePaths);
      const segmentLengths = studioTimeline.map(item => item.segmentLength).join('\n');
      const { task_id } = await startWorkflowGeneration({
        workflow_id: 'fflf_auto_v2',
        inputs: {
          images: folder_path,
          positive_prompt: studioMasterPrompt,
          negative_prompt: studioNegPrompt,
          segment_lengths: segmentLengths,
          initial_width: studioInitialWidth,
          upscale_factor: studioUpscaleFactor,
          seed: studioVideoSeed,
          looping: studioLooping,
        },
      });
      pollStudioTask(task_id);
    } catch (err) {
      setStudioIsGenerating(false);
      setStudioStatus(`Error: ${err.message}`);
    }
  };

  // Studio InfiniTalk generation
  const studioHandleInfiniTalk = async () => {
    const charImage = studioTimeline[0];
    if (!charImage) return alert(t('studioNeedImage'));
    if (!studioNarrationAudioPath) return alert(t('studioNeedAudio'));
    setStudioIsGenerating(true);
    setStudioProgress(0);
    setStudioStatus('Starting InfiniTalk...');
    try {
      const { task_id } = await startWorkflowGeneration({
        workflow_id: 'wan_infinitalk',
        inputs: {
          image: charImage.imagePath,
          audio: studioNarrationAudioPath,
          prompt: studioMasterPrompt,
          length: studioNarrationLength,
          width: 832,
          height: 480,
        },
      });
      pollStudioTask(task_id);
    } catch (err) {
      setStudioIsGenerating(false);
      setStudioStatus(`Error: ${err.message}`);
    }
  };

  // Studio Change Character generation
  const studioHandleChangeCharacter = async () => {
    const charImage = studioTimeline[0];
    if (!charImage) return alert(t('studioNeedImage'));
    if (!studioRefVideoPath) return alert(t('studioNeedRefVideo'));
    setStudioIsGenerating(true);
    setStudioProgress(0);
    setStudioStatus('Starting Change Character...');
    try {
      const { task_id } = await startWorkflowGeneration({
        workflow_id: 'change_character',
        inputs: {
          ref_image: charImage.imagePath,
          ref_video: studioRefVideoPath,
          prompt: studioScenePrompt,
          aspect_ratio: studioFluxAspectRatio,
        },
      });
      pollStudioTask(task_id);
    } catch (err) {
      setStudioIsGenerating(false);
      setStudioStatus(`Error: ${err.message}`);
    }
  };

  // Studio task polling (shared)
  const pollStudioTask = (taskId) => {
    const poll = setInterval(async () => {
      try {
        const s = await getTaskStatus(taskId);
        setStudioProgress(Math.round((s.progress || 0) * 100));
        setStudioStatus(s.message || '');
        if (s.status === 'completed') {
          clearInterval(poll);
          setStudioIsGenerating(false);
          setStudioOutputVideo(s.output_path || s.output_url);
        } else if (s.status === 'failed') {
          clearInterval(poll);
          setStudioIsGenerating(false);
          setStudioStatus(`Failed: ${s.message}`);
        }
      } catch { clearInterval(poll); setStudioIsGenerating(false); }
    }, 2000);
  };

  // Studio TTS generate
  const studioHandleTTSGenerate = async () => {
    if (!studioTtsScript.trim()) return;
    setStudioTtsGenerating(true);
    try {
      const { generateTTS } = await import('./api');
      const { audio_path, audio_url, duration, frame_count } = await generateTTS(
        studioTtsScript, studioTtsLanguage, studioTtsSpeaker
      );
      setStudioNarrationAudioPath(audio_path);
      setStudioNarrationAudio(audio_url);
      setStudioNarrationLength(frame_count);
    } catch (err) {
      alert(`TTS failed: ${err.message}`);
    } finally {
      setStudioTtsGenerating(false);
    }
  };

  // Studio YouTube download for Change Character
  const studioHandleYoutubeDownload = async () => {
    if (!studioYoutubeUrl.trim()) return;
    setStudioYoutubeDownloading(true);
    try {
      const result = await downloadYoutube(studioYoutubeUrl);
      setStudioRefVideoPath(result.path);
      setStudioRefVideoPreview(result.url || result.path);
    } catch (err) {
      alert(`Download failed: ${err.message}`);
    } finally {
      setStudioYoutubeDownloading(false);
    }
  };

  // Studio ref video upload handler
  const studioHandleRefVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const result = await uploadVideo(file);
      setStudioRefVideoPath(result.path);
      setStudioRefVideoPreview(URL.createObjectURL(file));
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    }
  };

  // Studio narration audio upload
  const studioHandleAudioUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const result = await uploadAudio(file);
      setStudioNarrationAudioPath(result.path);
      setStudioNarrationAudio(URL.createObjectURL(file));
      // Estimate frame count from duration (rough)
      if (result.duration) {
        setStudioNarrationLength(Math.round(result.duration * 20));
      }
    } catch (err) {
      alert(`Upload failed: ${err.message}`);
    }
  };

  // Studio chat send (Gemini AI)
  const studioHandleChatSend = async () => {
    if (!chatInput.trim() || chatLoading) return;
    const userMsg = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setChatLoading(true);
    try {
      const { sendStudioChat } = await import('./api');
      const { reply, actions } = await sendStudioChat(userMsg, chatHistory);
      setChatHistory(prev => [
        ...prev,
        { role: 'user', parts: [{ text: userMsg }] },
        { role: 'model', parts: [{ text: reply }] },
      ]);
      setChatMessages(prev => [...prev, { role: 'assistant', text: reply, actions }]);
      (actions || []).forEach(action => {
        if (action.tool === 'generate_idol_image' && action.result?.output_path) {
          studioAddToTimeline(action.result.output_path, action.result.absolute_path || action.result.output_path);
        }
      });
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', text: `Error: ${err.message}` }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Update prompts when language changes
  useEffect(() => {
    setI2vPrompt(translations[lang].i2vPromptDefault);
    setI2vNegPrompt(translations[lang].i2vNegPromptDefault);
  }, [lang]);

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
        // Auto-select portrait/landscape based on image aspect ratio
        setI2vResolution(result.height >= result.width ? '1280*720' : '720*1280');
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
      setI2vResolution(img.height >= img.width ? '1280*720' : '720*1280');
    }
    setShowI2vImagePicker(false);
  };

  // I2V generated output picker
  const toggleI2vOutputPicker = async () => {
    if (!showI2vOutputPicker) {
      try {
        const data = await listOutputs();
        setI2vGeneratedImages((data.outputs || []).filter(o => o.type === 'image'));
      } catch (err) { console.error(err); }
    }
    setShowI2vOutputPicker(!showI2vOutputPicker);
  };

  const selectI2vGeneratedImage = (img) => {
    setI2vImagePath(img.path); setI2vImagePreview(img.url); setI2vImageFile({ name: img.filename });
    if (img.width && img.height) {
      setI2vImageDimensions({ width: img.width, height: img.height });
      setI2vResolution(img.height >= img.width ? '1280*720' : '720*1280');
    }
    setShowI2vOutputPicker(false);
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
        aspect_ratio: fluxAspectRatio,
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

  // ─── Generic Workflow handlers ───
  const handleWfImageUpload = async (wfId, key, e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    updateWfPreview(wfId, key, URL.createObjectURL(file));
    try {
      const data = await uploadImage(file);
      updateWfFilePath(wfId, key, data.path);
    } catch (err) {
      updateWfState(wfId, { status: `Upload error: ${err.message}` });
    }
  };

  const handleWfBgImageUpload = async (wfId, key, e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    updateWfPreview(wfId, key, URL.createObjectURL(file));
    try {
      const data = await uploadBackground(file);
      updateWfFilePath(wfId, key, data.path);
    } catch (err) {
      updateWfState(wfId, { status: `Upload error: ${err.message}` });
    }
  };

  const handleWfBgGalleryToggle = async (wfId, key) => {
    const wfState = workflowStates[wfId];
    if (!wfState.bgGalleryOpen[key]) {
      try {
        const data = await listBackgrounds();
        updateWfState(wfId, { bgGalleryImages: data.backgrounds || [] });
      } catch {}
    }
    updateWfState(wfId, {
      bgGalleryOpen: { ...wfState.bgGalleryOpen, [key]: !wfState.bgGalleryOpen[key] },
    });
  };

  const handleWfAvatarInit = async (wfId) => {
    const wfState = workflowStates[wfId];
    if (wfState?.avatarGroups?.length > 0) return;
    try {
      const data = await listAvatarGroups();
      const groups = data.groups || [];
      updateWfState(wfId, { avatarGroups: groups });
      if (groups.length > 0) {
        handleWfAvatarSelect(wfId, groups[0]);
      }
    } catch {}
  };

  const handleWfAvatarSelect = async (wfId, group) => {
    setWorkflowStates(prev => ({
      ...prev, [wfId]: {
        ...prev[wfId],
        avatarSelectedGroup: { ...prev[wfId].avatarSelectedGroup, _current: group },
      },
    }));
    try {
      const data = await listAvatarImages(group);
      setWorkflowStates(prev => ({
        ...prev, [wfId]: {
          ...prev[wfId],
          avatarImages: { ...prev[wfId].avatarImages, [group]: data.images || [] },
        },
      }));
    } catch {}
  };

  const handleWfFashionInit = async (wfId) => {
    const wfState = workflowStates[wfId];
    if (wfState?.fashionStyles?.length > 0) return;
    try {
      const data = await getFashionStyles();
      updateWfState(wfId, {
        fashionStyles: data.styles || [],
        fashionCategories: data.categories || [],
      });
    } catch {}
  };

  const handleWfFashionApply = (wfId, prompt) => {
    setWorkflowStates(prev => ({
      ...prev, [wfId]: {
        ...prev[wfId],
        inputs: { ...prev[wfId].inputs, fashion_prompt: prompt },
      },
    }));
  };

  const handleWfFashionRandom = (wfId) => {
    const wfState = workflowStates[wfId];
    const styles = wfState?.fashionStyles || [];
    const cat = wfState?.fashionFilterCat || 'All';
    const filtered = cat === 'All' ? styles : styles.filter(s => s.category === cat);
    if (filtered.length === 0) return;
    const pick = filtered[Math.floor(Math.random() * filtered.length)];
    handleWfFashionApply(wfId, pick.prompt);
  };

  const handleWfVideoUpload = async (wfId, key, e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    updateWfPreview(wfId, key, URL.createObjectURL(file));
    try {
      const data = await uploadVideo(file);
      updateWfFilePath(wfId, key, data.path);
    } catch (err) {
      updateWfState(wfId, { status: `Upload error: ${err.message}` });
    }
  };

  const handleWfAudioUpload = async (wfId, key, e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    updateWfPreview(wfId, key, URL.createObjectURL(file));
    try {
      const data = await uploadAudio(file);
      updateWfFilePath(wfId, key, data.path);
    } catch (err) {
      updateWfState(wfId, { status: `Upload error: ${err.message}` });
    }
  };

  const handleWfYoutubeDownload = async (wfId, key) => {
    const wfState = workflowStates[wfId];
    const url = wfState?.youtubeUrl?.[key]?.trim();
    if (!url) return;
    updateWfState(wfId, {
      youtubeDownloading: { ...wfState.youtubeDownloading, [key]: true },
      status: t('wfDownloading'),
    });
    try {
      const data = await downloadYoutube(url);
      updateWfFilePath(wfId, key, data.path);
      updateWfPreview(wfId, key, data.url || `/api/uploads/video/${data.path.split('/').pop()}`);
      updateWfState(wfId, { status: '' });
    } catch (err) {
      updateWfState(wfId, { status: `Download error: ${err.message}` });
    } finally {
      setWorkflowStates(prev => ({
        ...prev, [wfId]: {
          ...prev[wfId],
          youtubeDownloading: { ...prev[wfId].youtubeDownloading, [key]: false },
        },
      }));
    }
  };

  // ---- Timeline helpers ----
  const generateRulerTicks = (duration) => {
    if (!duration || duration <= 0) return [];
    let interval;
    if (duration <= 10) interval = 1;
    else if (duration <= 30) interval = 2;
    else if (duration <= 60) interval = 5;
    else if (duration <= 300) interval = 10;
    else interval = 30;
    const ticks = [];
    for (let t = 0; t <= duration; t += interval) ticks.push(t);
    if (ticks[ticks.length - 1] !== Math.floor(duration)) ticks.push(duration);
    return ticks;
  };

  const formatTime = (seconds) => {
    if (seconds == null || isNaN(seconds)) return '0s';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.round((seconds % 1) * 10);
    return m > 0 ? `${m}:${String(s).padStart(2, '0')}.${ms}` : `${s}.${ms}s`;
  };

  const handleTimelineDragStart = useCallback((wfId, key, handleType, e) => {
    e.preventDefault();
    e.stopPropagation();
    const timelineEl = timelineRefsMap.current[`${wfId}-${key}`];
    if (!timelineEl) return;

    const onMouseMove = (moveEvent) => {
      const rect = timelineEl.getBoundingClientRect();
      const x = Math.max(0, Math.min(moveEvent.clientX - rect.left, rect.width));
      setWorkflowStates(prev => {
        const wf = prev[wfId];
        const duration = wf.videoDuration?.[key] || 0;
        if (!duration) return prev;
        const time = Math.round(((x / rect.width) * duration) * 100) / 100;
        const currentStart = parseFloat(wf.trimStart?.[key]) || 0;
        const currentEnd = parseFloat(wf.trimEnd?.[key]) || duration;
        let newStart = currentStart, newEnd = currentEnd;
        if (handleType === 'start') {
          newStart = Math.max(0, Math.min(time, currentEnd - 0.1));
        } else {
          newEnd = Math.min(duration, Math.max(time, currentStart + 0.1));
        }
        const videoEl = videoRefsMap.current[`${wfId}-${key}`];
        if (videoEl) videoEl.currentTime = handleType === 'start' ? newStart : newEnd;
        return {
          ...prev, [wfId]: {
            ...wf,
            trimStart: { ...wf.trimStart, [key]: newStart },
            trimEnd: { ...wf.trimEnd, [key]: newEnd },
            playheadPosition: { ...wf.playheadPosition, [key]: handleType === 'start' ? newStart : newEnd },
          },
        };
      });
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, []);

  const handleRulerClick = useCallback((wfId, key, e) => {
    const timelineEl = timelineRefsMap.current[`${wfId}-${key}`];
    if (!timelineEl) return;
    const rect = timelineEl.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setWorkflowStates(prev => {
      const wf = prev[wfId];
      const duration = wf.videoDuration?.[key] || 0;
      if (!duration) return prev;
      const time = Math.round(((x / rect.width) * duration) * 100) / 100;
      const videoEl = videoRefsMap.current[`${wfId}-${key}`];
      if (videoEl) videoEl.currentTime = time;
      return {
        ...prev, [wfId]: {
          ...wf,
          playheadPosition: { ...wf.playheadPosition, [key]: time },
        },
      };
    });
  }, []);

  const handleVideoTimeUpdate = useCallback((wfId, key, e) => {
    const currentTime = Math.round(e.target.currentTime * 100) / 100;
    setWorkflowStates(prev => ({
      ...prev, [wfId]: {
        ...prev[wfId],
        playheadPosition: { ...prev[wfId].playheadPosition, [key]: currentTime },
      },
    }));
  }, []);

  const handleWfVideoTrim = async (wfId, key) => {
    const wfState = workflowStates[wfId];
    if (!wfState?.filePaths[key]) return;
    const start = parseFloat(wfState.trimStart?.[key]) || 0;
    const end = parseFloat(wfState.trimEnd?.[key]) || wfState.videoDuration?.[key] || 0;
    if (end <= start) return;
    setWorkflowStates(prev => ({
      ...prev, [wfId]: { ...prev[wfId], trimming: { ...prev[wfId].trimming, [key]: true } },
    }));
    try {
      const result = await trimVideo(wfState.filePaths[key], start, end);
      updateWfFilePath(wfId, key, result.path);
      updateWfPreview(wfId, key, result.url);
      setWorkflowStates(prev => ({
        ...prev, [wfId]: {
          ...prev[wfId],
          videoDuration: { ...prev[wfId].videoDuration, [key]: result.duration },
          trimStart: { ...prev[wfId].trimStart, [key]: 0 },
          trimEnd: { ...prev[wfId].trimEnd, [key]: result.duration },
          trimming: { ...prev[wfId].trimming, [key]: false },
          status: '',
        },
      }));
    } catch (err) {
      setWorkflowStates(prev => ({
        ...prev, [wfId]: {
          ...prev[wfId],
          trimming: { ...prev[wfId].trimming, [key]: false },
          status: `Trim error: ${err.message}`,
        },
      }));
    }
  };

  const handleWfGalleryToggle = async (wfId, key) => {
    const wfState = workflowStates[wfId];
    if (!wfState.galleryOpen[key]) {
      try {
        const [uploadData, outputData] = await Promise.all([
          listUploadedImages().catch(() => ({ images: [] })),
          listOutputs().catch(() => ({ outputs: [] })),
        ]);
        const uploads = (uploadData.images || []).map(img => ({ ...img, source: 'upload' }));
        const outputs = (outputData.outputs || [])
          .filter(o => o.type === 'image')
          .map(o => ({ url: o.url, path: o.path, source: 'output' }));
        const allImages = [...outputs, ...uploads];
        updateWfState(wfId, { galleryImages: allImages });
        setUploadedImages(uploadData.images || []);
      } catch {}
    }
    updateWfState(wfId, {
      galleryOpen: { ...wfState.galleryOpen, [key]: !wfState.galleryOpen[key] },
    });
  };

  const handleWfGallerySelect = (wfId, key, img) => {
    setWorkflowStates(prev => {
      const wf = prev[wfId];
      const current = wf.gallerySelected[key] || [];
      const exists = current.some(i => i.path === img.path);
      const updated = exists ? current.filter(i => i.path !== img.path) : [...current, img];
      return {
        ...prev, [wfId]: {
          ...wf,
          gallerySelected: { ...wf.gallerySelected, [key]: updated },
          filePaths: { ...wf.filePaths, [key]: updated.map(i => i.path) },
          previews: { ...wf.previews, [key]: updated.map(i => i.url) },
        },
      };
    });
  };

  const handleWfGenerate = async (wfId) => {
    const wfState = workflowStates[wfId];
    const wfDef = workflows.find(w => w.id === wfId);
    if (!wfDef || !wfState) return;

    // Validate required inputs
    for (const inp of wfDef.inputs) {
      if (inp.required) {
        const val = (inp.type === 'image' || inp.type === 'video')
          ? wfState.filePaths[inp.key]
          : inp.type === 'gallery_select'
            ? (wfState.gallerySelected[inp.key]?.length > 0)
            : wfState.inputs[inp.key];
        if (!val) {
          updateWfState(wfId, { status: `Required: ${inp.label[lang] || inp.key}` });
          return;
        }
      }
    }

    updateWfState(wfId, { isGenerating: true, progress: 0, outputVideo: null, status: t('wfGenerating') });

    // Build inputs payload
    const payload = { ...wfState.inputs };
    for (const [key, path] of Object.entries(wfState.filePaths)) {
      if (Array.isArray(path)) continue; // gallery_select handled below
      payload[key] = path;
    }

    // Handle gallery_select: prepare images folder
    for (const inp of wfDef.inputs) {
      if (inp.type === 'gallery_select') {
        const selectedPaths = (wfState.gallerySelected[inp.key] || []).map(i => i.path);
        if (selectedPaths.length > 0) {
          try {
            const result = await prepareWorkflowImages(selectedPaths);
            payload[inp.key] = result.folder_path;
          } catch (err) {
            updateWfState(wfId, { isGenerating: false, status: `Prepare images error: ${err.message}` });
            return;
          }
        }
      }
    }

    try {
      const data = await startWorkflowGeneration({
        workflow_id: wfId, inputs: payload,
        yt_title: wfState.ytTitle || '',
        yt_description: wfState.ytDescription || '',
        yt_hashtags: wfState.ytHashtags || '',
      });
      const taskId = data.task_id;
      updateWfState(wfId, { currentTaskId: taskId });
      const poll = setInterval(async () => {
        try {
          const s = await getTaskStatus(taskId);
          updateWfState(wfId, {
            progress: Math.round((s.progress || 0) * 100),
            status: s.status_message || s.status,
          });
          if (s.status === 'completed') {
            clearInterval(poll);
            updateWfState(wfId, { isGenerating: false, progress: 100, outputVideo: s.output_url, currentTaskId: null });
          } else if (s.status === 'failed') {
            clearInterval(poll);
            updateWfState(wfId, { isGenerating: false, status: `Error: ${s.error || 'Failed'}`, currentTaskId: null });
          } else if (s.status === 'cancelled') {
            clearInterval(poll);
            updateWfState(wfId, { isGenerating: false, status: t('wfCancelled'), currentTaskId: null });
          }
        } catch (err) {
          clearInterval(poll);
          updateWfState(wfId, { isGenerating: false, status: `Polling error: ${err.message}`, currentTaskId: null });
        }
      }, 3000);
    } catch (err) {
      updateWfState(wfId, { isGenerating: false, status: `Error: ${err.message}` });
    }
  };

  const handleWfCancel = async (wfId) => {
    const wfState = workflowStates[wfId];
    const taskId = wfState?.currentTaskId;
    if (!taskId) return;
    try {
      await cancelGeneration(taskId);
    } catch {}
    updateWfState(wfId, { isGenerating: false, status: t('wfCancelled'), currentTaskId: null });
  };

  // ─── Queue handlers ───
  // Keep ref in sync for use inside async loops
  useEffect(() => { wfQueueRef.current = wfQueue; }, [wfQueue]);

  const updateQueueItem = useCallback((wfId, itemId, updates) => {
    setWfQueue(prev => {
      const q = prev[wfId];
      if (!q) return prev;
      return {
        ...prev,
        [wfId]: {
          ...q,
          items: q.items.map(it => it.id === itemId ? { ...it, ...updates } : it),
        },
      };
    });
  }, []);

  const handleWfQueueAdd = (wfId) => {
    const wfState = workflowStates[wfId];
    const wfDef = workflows.find(w => w.id === wfId);
    if (!wfDef || !wfState) return;

    // Validate required inputs
    for (const inp of wfDef.inputs) {
      if (inp.required) {
        const val = (inp.type === 'image' || inp.type === 'video')
          ? wfState.filePaths[inp.key]
          : inp.type === 'gallery_select'
            ? (wfState.gallerySelected[inp.key]?.length > 0)
            : wfState.inputs[inp.key];
        if (!val) {
          updateWfState(wfId, { status: `Required: ${inp.label[lang] || inp.key}` });
          return;
        }
      }
    }

    const item = {
      id: crypto.randomUUID(),
      label: `Job ${(wfQueue[wfId]?.items?.length || 0) + 1}`,
      inputs: { ...wfState.inputs },
      filePaths: { ...wfState.filePaths },
      previews: { ...wfState.previews },
      gallerySelected: { ...wfState.gallerySelected },
      ytTitle: wfState.ytTitle || '',
      ytDescription: wfState.ytDescription || '',
      ytHashtags: wfState.ytHashtags || '',
      status: 'pending',
      progress: 0,
      outputVideo: null,
      error: null,
    };

    setWfQueue(prev => ({
      ...prev,
      [wfId]: {
        ...prev[wfId],
        items: [...(prev[wfId]?.items || []), item],
        isProcessing: prev[wfId]?.isProcessing || false,
      },
    }));
    updateWfState(wfId, { status: `Added: ${item.label}` });
  };

  const handleWfQueueStart = async (wfId) => {
    const queue = wfQueue[wfId];
    if (!queue?.items?.length || queue.isProcessing) return;

    setWfQueue(prev => ({ ...prev, [wfId]: { ...prev[wfId], isProcessing: true } }));

    const wfDef = workflows.find(w => w.id === wfId);
    // Get pending items from current state snapshot
    const pendingIds = queue.items.filter(i => i.status === 'pending').map(i => i.id);

    for (const itemId of pendingIds) {
      // Read latest item from ref
      const currentItem = wfQueueRef.current[wfId]?.items?.find(i => i.id === itemId);
      if (!currentItem || currentItem.status !== 'pending') continue;

      updateQueueItem(wfId, itemId, { status: 'running', progress: 0 });

      try {
        // Build payload
        const payload = { ...currentItem.inputs };
        for (const [key, path] of Object.entries(currentItem.filePaths)) {
          if (!Array.isArray(path)) payload[key] = path;
        }
        for (const inp of wfDef.inputs) {
          if (inp.type === 'gallery_select') {
            const sel = (currentItem.gallerySelected[inp.key] || []).map(i => i.path);
            if (sel.length > 0) {
              const result = await prepareWorkflowImages(sel);
              payload[inp.key] = result.folder_path;
            }
          }
        }

        const data = await startWorkflowGeneration({
          workflow_id: wfId, inputs: payload,
          yt_title: currentItem.ytTitle || '',
          yt_description: currentItem.ytDescription || '',
          yt_hashtags: currentItem.ytHashtags || '',
        });
        const taskId = data.task_id;

        // Poll until done
        await new Promise((resolve) => {
          const poll = setInterval(async () => {
            try {
              const s = await getTaskStatus(taskId);
              updateQueueItem(wfId, itemId, {
                progress: Math.round((s.progress || 0) * 100),
              });
              if (s.status === 'completed') {
                clearInterval(poll);
                updateQueueItem(wfId, itemId, {
                  status: 'completed', progress: 100,
                  outputVideo: s.output_url || s.output_path,
                });
                resolve();
              } else if (s.status === 'failed') {
                clearInterval(poll);
                updateQueueItem(wfId, itemId, {
                  status: 'failed', error: s.message || 'Failed',
                });
                resolve();
              } else if (s.status === 'cancelled') {
                clearInterval(poll);
                updateQueueItem(wfId, itemId, {
                  status: 'failed', error: 'Cancelled',
                });
                resolve();
              }
            } catch (err) {
              clearInterval(poll);
              updateQueueItem(wfId, itemId, { status: 'failed', error: err.message });
              resolve();
            }
          }, 3000);
        });
      } catch (err) {
        updateQueueItem(wfId, itemId, { status: 'failed', error: err.message });
      }
    }

    setWfQueue(prev => ({ ...prev, [wfId]: { ...prev[wfId], isProcessing: false } }));
  };

  const handleWfQueueRemove = (wfId, itemId) => {
    setWfQueue(prev => {
      const q = prev[wfId];
      if (!q) return prev;
      return { ...prev, [wfId]: { ...q, items: q.items.filter(i => i.id !== itemId || i.status === 'running') } };
    });
  };

  const handleWfQueueClear = (wfId) => {
    setWfQueue(prev => {
      const q = prev[wfId];
      if (!q) return prev;
      return { ...prev, [wfId]: { ...q, items: q.items.filter(i => i.status === 'pending' || i.status === 'running') } };
    });
  };

  // ─── Dynamic form renderer ───
  const renderWorkflowInput = (wfId, inputDef) => {
    const wfState = workflowStates[wfId];
    if (!wfState) return null;
    const label = inputDef.label?.[lang] || inputDef.label?.en || inputDef.key;
    const inputIdBase = `wf-${wfId}-${inputDef.key}`;

    switch (inputDef.type) {
      case 'image': {
        if (inputDef.avatar_gallery) {
          const groups = wfState.avatarGroups || [];
          const currentGroup = wfState.avatarSelectedGroup?._current || '';
          const avatarImgs = wfState.avatarImages?.[currentGroup] || [];
          if (groups.length === 0) handleWfAvatarInit(wfId);
          return (
            <div className="card" key={inputDef.key}>
              <h3>{label}</h3>
              {wfState.previews[inputDef.key] && (
                <div className="avatar-viewer">
                  <img src={wfState.previews[inputDef.key]} alt="" />
                </div>
              )}
              <div className="avatar-selector">
                <select className="avatar-dropdown" value={currentGroup}
                  onChange={e => handleWfAvatarSelect(wfId, e.target.value)}>
                  {groups.map(g => <option key={g} value={g}>{g}</option>)}
                </select>
              </div>
              {avatarImgs.length > 0 && (
                <div className="avatar-thumbs">
                  {avatarImgs.map((img, i) => (
                    <div key={i} className={`avatar-thumb-item${wfState.filePaths[inputDef.key] === img.path ? ' selected' : ''}`}
                      onClick={() => {
                        updateWfFilePath(wfId, inputDef.key, img.path);
                        updateWfPreview(wfId, inputDef.key, img.url);
                      }}>
                      <img src={img.url} alt={img.filename} />
                      <span className="avatar-thumb-name">{img.filename.replace(/\.[^.]+$/, '')}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        }
        if (inputDef.background_gallery) {
          return (
            <div className="card" key={inputDef.key}>
              <h3>{label}</h3>
              <div className="drop-zone" onClick={() => document.getElementById(inputIdBase)?.click()}
                onDragOver={e => e.preventDefault()}
                onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) { const dt = new DataTransfer(); dt.items.add(f); const inp = document.getElementById(inputIdBase); inp.files = dt.files; inp.dispatchEvent(new Event('change', { bubbles: true })); } }}>
                {wfState.previews[inputDef.key]
                  ? <img src={wfState.previews[inputDef.key]} alt="" style={{ maxHeight: 200, objectFit: 'contain' }} />
                  : <p>{t('dropImageHere')}</p>}
              </div>
              <input id={inputIdBase} type="file" accept="image/*" style={{ display: 'none' }}
                onChange={e => handleWfBgImageUpload(wfId, inputDef.key, e)} />
              <button className="btn secondary" style={{ marginTop: 8 }} onClick={() => handleWfBgGalleryToggle(wfId, inputDef.key)}>
                {t('wfBgGallery')}
              </button>
              {wfState.bgGalleryOpen[inputDef.key] && (
                <div className="bg-gallery-grid">
                  {(wfState.bgGalleryImages || []).length === 0 && (
                    <p className="bg-gallery-empty">{t('wfBgGalleryEmpty')}</p>
                  )}
                  {(wfState.bgGalleryImages || []).map((img, i) => (
                    <div key={i} className={`bg-gallery-item${wfState.filePaths[inputDef.key] === img.path ? ' selected' : ''}`}
                      onClick={() => {
                        updateWfFilePath(wfId, inputDef.key, img.path);
                        updateWfPreview(wfId, inputDef.key, img.url);
                      }}>
                      <img src={img.url} alt={img.filename} />
                      <span className="bg-gallery-name">{img.filename}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        }
        return (
          <div className="card" key={inputDef.key}>
            <h3>{label}</h3>
            <div className="drop-zone" onClick={() => document.getElementById(inputIdBase)?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) { const dt = new DataTransfer(); dt.items.add(f); const inp = document.getElementById(inputIdBase); inp.files = dt.files; inp.dispatchEvent(new Event('change', { bubbles: true })); } }}>
              {wfState.previews[inputDef.key]
                ? <img src={wfState.previews[inputDef.key]} alt="" style={{ maxHeight: 200, objectFit: 'contain' }} />
                : <p>{t('dropImageHere')}</p>}
            </div>
            <input id={inputIdBase} type="file" accept="image/*" style={{ display: 'none' }} onChange={e => handleWfImageUpload(wfId, inputDef.key, e)} />
            <button className="btn secondary" style={{ marginTop: 8 }} onClick={() => handleWfGalleryToggle(wfId, inputDef.key)}>
              {t('wfSelectImages') || t('selectFromUploads')}
            </button>
            {wfState.galleryOpen[inputDef.key] && (
              <div className="uploaded-images-grid" style={{ marginTop: 8 }}>
                {(wfState.galleryImages || uploadedImages).map((img, i) => (
                  <img key={i} src={img.url} alt="" className="uploaded-thumb" onClick={() => { updateWfFilePath(wfId, inputDef.key, img.path); updateWfPreview(wfId, inputDef.key, img.url); updateWfState(wfId, { galleryOpen: { ...wfState.galleryOpen, [inputDef.key]: false } }); }} />
                ))}
              </div>
            )}
          </div>
        );
      }

      case 'audio': {
        return (
          <div className="card" key={inputDef.key}>
            <h3>{label}</h3>
            <div className="drop-zone" onClick={() => document.getElementById(inputIdBase)?.click()}>
              {wfState.previews[inputDef.key]
                ? <audio src={wfState.previews[inputDef.key]} controls style={{ width: '100%' }} />
                : <p>{t('dropAudioHere') || 'Drop audio file here or click to browse'}</p>}
            </div>
            <input id={inputIdBase} type="file" accept="audio/*" style={{ display: 'none' }} onChange={e => handleWfAudioUpload(wfId, inputDef.key, e)} />
          </div>
        );
      }

      case 'video': {
        const videoMode = wfState.videoInputMode?.[inputDef.key] || 'upload';
        const hasVideo = !!wfState.filePaths[inputDef.key];
        const duration = wfState.videoDuration?.[inputDef.key];
        const isTrimming = wfState.trimming?.[inputDef.key];
        const trimStartVal = parseFloat(wfState.trimStart?.[inputDef.key]) || 0;
        const trimEndVal = parseFloat(wfState.trimEnd?.[inputDef.key]) || duration || 0;
        const playhead = wfState.playheadPosition?.[inputDef.key] || 0;
        const refKey = `${wfId}-${inputDef.key}`;

        const videoMetaHandler = (e) => {
          const dur = e.target.duration;
          if (dur && isFinite(dur)) {
            setWorkflowStates(prev => ({
              ...prev, [wfId]: {
                ...prev[wfId],
                videoDuration: { ...prev[wfId].videoDuration, [inputDef.key]: Math.round(dur * 100) / 100 },
                trimStart: { ...prev[wfId].trimStart, [inputDef.key]: prev[wfId].trimStart?.[inputDef.key] ?? 0 },
                trimEnd: { ...prev[wfId].trimEnd, [inputDef.key]: prev[wfId].trimEnd?.[inputDef.key] ?? Math.round(dur * 100) / 100 },
              },
            }));
          }
        };

        return (
          <div className="card video-editor-card" key={inputDef.key}>
            <h3>{label}</h3>
            {inputDef.allow_youtube && (
              <div className="video-input-tabs">
                <button className={`tab-btn${videoMode === 'upload' ? ' active' : ''}`}
                  onClick={() => updateWfState(wfId, { videoInputMode: { ...wfState.videoInputMode, [inputDef.key]: 'upload' } })}>
                  {t('wfVideoUpload')}
                </button>
                <button className={`tab-btn${videoMode === 'youtube' ? ' active' : ''}`}
                  onClick={() => updateWfState(wfId, { videoInputMode: { ...wfState.videoInputMode, [inputDef.key]: 'youtube' } })}>
                  {t('wfVideoYoutube')}
                </button>
              </div>
            )}

            <div className="video-editor-layout">
              {/* LEFT: Video Viewer + Timeline */}
              <div className="video-editor-viewer">
                {videoMode === 'upload' && (
                  <>
                    <div className="drop-zone video-drop-zone" onClick={() => document.getElementById(inputIdBase)?.click()}
                      onDragOver={e => e.preventDefault()}
                      onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) { const dt = new DataTransfer(); dt.items.add(f); const inp = document.getElementById(inputIdBase); inp.files = dt.files; inp.dispatchEvent(new Event('change', { bubbles: true })); } }}>
                      {wfState.previews[inputDef.key]
                        ? <video ref={el => { videoRefsMap.current[refKey] = el; }}
                            src={wfState.previews[inputDef.key]} className="video-editor-player" controls muted
                            onLoadedMetadata={videoMetaHandler}
                            onTimeUpdate={e => handleVideoTimeUpdate(wfId, inputDef.key, e)} />
                        : <p>{t('dropVideoHere')}</p>}
                    </div>
                    <input id={inputIdBase} type="file" accept="video/*" style={{ display: 'none' }} onChange={e => handleWfVideoUpload(wfId, inputDef.key, e)} />
                  </>
                )}
                {videoMode === 'youtube' && (
                  <div className="youtube-input">
                    <input type="text" placeholder="YouTube URL"
                      value={wfState.youtubeUrl?.[inputDef.key] || ''}
                      onChange={e => updateWfState(wfId, { youtubeUrl: { ...wfState.youtubeUrl, [inputDef.key]: e.target.value } })}
                      disabled={wfState.youtubeDownloading?.[inputDef.key]} />
                    <button className="btn secondary" onClick={() => handleWfYoutubeDownload(wfId, inputDef.key)}
                      disabled={wfState.youtubeDownloading?.[inputDef.key] || !wfState.youtubeUrl?.[inputDef.key]?.trim()}>
                      {wfState.youtubeDownloading?.[inputDef.key] ? t('wfDownloading') : t('wfDownloadBtn')}
                    </button>
                    {wfState.previews[inputDef.key] && videoMode === 'youtube' && (
                      <video ref={el => { videoRefsMap.current[refKey] = el; }}
                        src={wfState.previews[inputDef.key]} className="video-editor-player" style={{ marginTop: 8 }} controls muted
                        onLoadedMetadata={videoMetaHandler}
                        onTimeUpdate={e => handleVideoTimeUpdate(wfId, inputDef.key, e)} />
                    )}
                  </div>
                )}

                {/* Timeline */}
                {hasVideo && duration > 0 && (
                  <div className="video-timeline-container">
                    {/* Ruler with ticks */}
                    <div className="video-timeline-ruler"
                      ref={el => { timelineRefsMap.current[refKey] = el; }}
                      onClick={e => handleRulerClick(wfId, inputDef.key, e)}>
                      {generateRulerTicks(duration).map(tick => (
                        <div key={tick} className="ruler-tick" style={{ left: `${(tick / duration) * 100}%` }}>
                          <div className="ruler-tick-line" />
                          <span className="ruler-tick-label">{formatTime(tick)}</span>
                        </div>
                      ))}
                      <div className="timeline-playhead" style={{ left: `${(playhead / duration) * 100}%` }} />
                    </div>
                    {/* Track with handles */}
                    <div className="video-timeline-track">
                      <div className="timeline-region-dimmed" style={{ left: 0, width: `${(trimStartVal / duration) * 100}%` }} />
                      <div className="timeline-region-active" style={{ left: `${(trimStartVal / duration) * 100}%`, width: `${((trimEndVal - trimStartVal) / duration) * 100}%` }} />
                      <div className="timeline-region-dimmed" style={{ left: `${(trimEndVal / duration) * 100}%`, width: `${((duration - trimEndVal) / duration) * 100}%` }} />
                      <div className="timeline-handle timeline-handle-left" style={{ left: `${(trimStartVal / duration) * 100}%` }}
                        onMouseDown={e => handleTimelineDragStart(wfId, inputDef.key, 'start', e)}>
                        <div className="handle-grip" />
                      </div>
                      <div className="timeline-handle timeline-handle-right" style={{ left: `${(trimEndVal / duration) * 100}%` }}
                        onMouseDown={e => handleTimelineDragStart(wfId, inputDef.key, 'end', e)}>
                        <div className="handle-grip" />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* RIGHT: Controls Panel */}
              {hasVideo && duration > 0 && (
                <div className="video-editor-controls">
                  <div className="video-edit-section">
                    <div className="video-edit-title">{t('wfVideoEdit')}</div>
                    <div className="trim-duration">{t('wfDuration')}: {formatTime(duration)}</div>
                    <div className="trim-controls-vertical">
                      <label>{t('wfTrimStart')}:
                        <input type="number" className="trim-input" min={0} max={duration} step={0.1}
                          value={trimStartVal}
                          onChange={e => {
                            const val = parseFloat(e.target.value) || 0;
                            setWorkflowStates(prev => ({
                              ...prev, [wfId]: { ...prev[wfId],
                                trimStart: { ...prev[wfId].trimStart, [inputDef.key]: val },
                                playheadPosition: { ...prev[wfId].playheadPosition, [inputDef.key]: val },
                              },
                            }));
                            const videoEl = videoRefsMap.current[refKey];
                            if (videoEl) videoEl.currentTime = val;
                          }} />
                        <span className="trim-unit">s</span>
                      </label>
                      <label>{t('wfTrimEnd')}:
                        <input type="number" className="trim-input" min={0} max={duration} step={0.1}
                          value={trimEndVal}
                          onChange={e => {
                            const val = parseFloat(e.target.value) || duration;
                            setWorkflowStates(prev => ({
                              ...prev, [wfId]: { ...prev[wfId],
                                trimEnd: { ...prev[wfId].trimEnd, [inputDef.key]: val },
                                playheadPosition: { ...prev[wfId].playheadPosition, [inputDef.key]: val },
                              },
                            }));
                            const videoEl = videoRefsMap.current[refKey];
                            if (videoEl) videoEl.currentTime = val;
                          }} />
                        <span className="trim-unit">s</span>
                      </label>
                    </div>
                    <div className="trim-selection-info">
                      {formatTime(trimStartVal)} - {formatTime(trimEndVal)} ({formatTime(trimEndVal - trimStartVal)})
                    </div>
                    <div className="trim-btn-row">
                      <button className="btn secondary trim-btn" onClick={() => handleWfVideoTrim(wfId, inputDef.key)}
                        disabled={isTrimming}>
                        {isTrimming ? t('wfTrimming') : t('wfTrimApply')}
                      </button>
                      <a href={wfState.previews[inputDef.key]} download className="btn secondary trim-btn">
                        {t('download')}
                      </a>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      }

      case 'text': {
        return (
          <div className="card" key={inputDef.key}>
            <h3>{label}</h3>
            <div className="form-group">
              <textarea value={wfState.inputs[inputDef.key] ?? inputDef.default ?? ''} onChange={e => updateWfInput(wfId, inputDef.key, e.target.value)} rows={inputDef.rows || 4} />
            </div>
          </div>
        );
      }

      case 'number': {
        return (
          <div className="card" key={inputDef.key}>
            <h3>{label}: {wfState.inputs[inputDef.key] ?? inputDef.default ?? 0}</h3>
            <div className="form-group">
              <input type="range" value={wfState.inputs[inputDef.key] ?? inputDef.default ?? 0}
                onChange={e => updateWfInput(wfId, inputDef.key, parseFloat(e.target.value))}
                min={inputDef.min} max={inputDef.max} step={inputDef.step} />
            </div>
          </div>
        );
      }

      case 'toggle': {
        return (
          <div className="card" key={inputDef.key}>
            <label className="toggle-label">
              <input type="checkbox" checked={wfState.inputs[inputDef.key] ?? inputDef.default ?? false}
                onChange={e => updateWfInput(wfId, inputDef.key, e.target.checked)} />
              <span>{label}</span>
            </label>
          </div>
        );
      }

      case 'select_buttons': {
        return (
          <div className="card" key={inputDef.key}>
            <h3>{label}</h3>
            <div className="aspect-ratio-btns">
              {inputDef.options.map(opt => (
                <button key={opt.value}
                  className={`btn${(wfState.inputs[inputDef.key] || inputDef.default) === opt.value ? '' : ' secondary'}`}
                  onClick={() => updateWfInput(wfId, inputDef.key, opt.value)}>
                  {opt.label?.[lang] || opt.label?.en || opt.value}
                </button>
              ))}
            </div>
          </div>
        );
      }

      case 'fashion_select': {
        if (!wfState.fashionStyles || wfState.fashionStyles.length === 0) {
          handleWfFashionInit(wfId);
        }
        const styles = wfState.fashionStyles || [];
        const categories = wfState.fashionCategories || [];
        const filterCat = wfState.fashionFilterCat || 'All';
        const filtered = filterCat === 'All' ? styles : styles.filter(s => s.category === filterCat);
        return (
          <div className="card" key={inputDef.key}>
            <h3>{t('fashionStyle')}</h3>
            <div className="fashion-filter-row">
              <button className={`fashion-cat-btn${filterCat === 'All' ? ' active' : ''}`}
                onClick={() => updateWfState(wfId, { fashionFilterCat: 'All' })}>
                {t('fashionAll')}
              </button>
              {categories.map(cat => (
                <button key={cat} className={`fashion-cat-btn${filterCat === cat ? ' active' : ''}`}
                  onClick={() => updateWfState(wfId, { fashionFilterCat: cat })}>
                  {cat}
                </button>
              ))}
              <button className="btn secondary fashion-random-btn" onClick={() => handleWfFashionRandom(wfId)}>
                {t('fashionRandom')}
              </button>
            </div>
            <div className="fashion-grid">
              {filtered.map(style => (
                <div key={style.id} className={`fashion-item${wfState.inputs?.fashion_prompt === style.prompt ? ' selected' : ''}`}
                  onClick={() => handleWfFashionApply(wfId, style.prompt)}>
                  <span className="fashion-item-cat">{style.category}</span>
                  <span className="fashion-item-text">{style.prompt}</span>
                </div>
              ))}
            </div>
          </div>
        );
      }

      case 'gallery_select': {
        const selected = wfState.gallerySelected?.[inputDef.key] || [];
        return (
          <div className="card" key={inputDef.key}>
            <h3>{label} {selected.length > 0 && `(${selected.length} ${t('wfSelectedCount')})`}</h3>
            {selected.length > 0 && (
              <div className="multi-image-preview">
                {selected.map((img, i) => (
                  <img key={i} src={img.url} alt="" style={{ maxHeight: 60, objectFit: 'contain', borderRadius: 4 }} />
                ))}
              </div>
            )}
            <button className="btn secondary" style={{ marginTop: 8 }} onClick={() => handleWfGalleryToggle(wfId, inputDef.key)}>
              {t('wfSelectImages')}
            </button>
            {wfState.galleryOpen?.[inputDef.key] && (
              <div className="gallery-select-grid" style={{ marginTop: 8 }}>
                {(wfState.galleryImages || uploadedImages).map((img, i) => (
                  <div key={i} className={`gallery-select-item${selected.some(s => s.path === img.path) ? ' selected' : ''}`}
                    onClick={() => handleWfGallerySelect(wfId, inputDef.key, img)}>
                    <img src={img.url} alt="" style={{ width: '100%', height: 80, objectFit: 'cover' }} />
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      }

      default:
        return null;
    }
  };

  // ─── Auth gate: loading ───
  if (authLoading) {
    return <div className="loading-page"><div className="loading-spinner" /></div>;
  }

  // ─── Auth gate: pending approval ───
  if (isPendingApproval) {
    return (
      <div className="pending-page">
        <div className="pending-card">
          <img src="/logo.png" alt="Logo" className="logo-img" />
          <h2>{t('loginTitle')}</h2>
          <p>{t('loginPending')}</p>
          <button className="login-btn" onClick={handleLogout}>{t('logoutBtn')}</button>
        </div>
      </div>
    );
  }

  // ─── Auth gate: login page ───
  if (!authUser) {
    return (
      <div className="login-page">
        <div className="login-card">
          <img src="/logo.png" alt="Logo" className="logo-img" />
          <h2>{t('loginTitle')}</h2>

          <button className="google-signin-btn" onClick={handleGoogleLogin} disabled={loginLoading}>
            <svg width="20" height="20" viewBox="0 0 48 48"><path fill="#EA4335" d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"/><path fill="#4285F4" d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"/><path fill="#FBBC05" d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"/><path fill="#34A853" d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"/></svg>
            {t('loginGoogleBtn')}
          </button>

          <div className="login-divider">OR</div>

          <form className="login-form" onSubmit={handleEmailLogin}>
            <input
              className="login-input"
              type="email"
              placeholder={t('loginEmail')}
              value={loginEmail}
              onChange={e => setLoginEmail(e.target.value)}
              required
            />
            <input
              className="login-input"
              type="password"
              placeholder={t('loginPassword')}
              value={loginPassword}
              onChange={e => setLoginPassword(e.target.value)}
              required
            />
            <label className="remember-email-label">
              <input
                type="checkbox"
                checked={rememberEmail}
                onChange={e => setRememberEmail(e.target.checked)}
              />
              {t('rememberEmail')}
            </label>
            <button className="login-btn" type="submit" disabled={loginLoading}>
              {loginLoading ? '...' : t('loginSignIn')}
            </button>
          </form>

          {loginError && <p className="login-error">{loginError}</p>}

          <div style={{ marginTop: '1rem' }}>
            <div className="language-selector" style={{ justifyContent: 'center' }}>
              <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>EN</button>
              <button className={lang === 'ko' ? 'active' : ''} onClick={() => setLang('ko')}>KO</button>
              <button className={lang === 'zh' ? 'active' : ''} onClick={() => setLang('zh')}>ZH</button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-layout">
      {/* Top Header */}
      <header className="top-header">
        <h1 className="logo"><img src="/logo.png" alt="Logo" className="logo-img" /></h1>
        <div className="header-user-area">
          <div className="language-selector">
            <button className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>EN</button>
            <button className={lang === 'ko' ? 'active' : ''} onClick={() => setLang('ko')}>KO</button>
            <button className={lang === 'zh' ? 'active' : ''} onClick={() => setLang('zh')}>ZH</button>
          </div>
          <div className="header-user-info">
            {authUser.picture && <img src={authUser.picture} alt="" className="header-user-avatar" referrerPolicy="no-referrer" />}
            <span className="header-user-name">{authUser.name || authUser.email}</span>
          </div>
          <button className="header-logout-btn" onClick={handleLogout}>{t('logoutBtn')}</button>
        </div>
      </header>

      <div className="app-body">
        {/* Sidebar */}
        <nav className="sidebar">
          <div
            className={`sidebar-item${activeMenu === 'studio' ? ' active' : ''}`}
            onClick={() => setActiveMenu('studio')}
          >
            <span className="sidebar-icon">&#127916;</span>
            {t('menuStudio')}
          </div>
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
            className={`sidebar-item${activeMenu === 'workflow' ? ' active' : ''}`}
            onClick={() => setActiveMenu('workflow')}
          >
            <span className="sidebar-icon">&#9881;</span>
            {t('menuWorkflow')}
          </div>
          <div
            className={`sidebar-item${activeMenu === 'gallery' ? ' active' : ''}`}
            onClick={() => setActiveMenu('gallery')}
          >
            <span className="sidebar-icon">&#128247;</span>
            {t('menuGallery')}
          </div>
          {authUser.role === 'superadmin' && (
            <div
              className={`sidebar-item${activeMenu === 'admin' ? ' active' : ''}`}
              onClick={() => setActiveMenu('admin')}
            >
              <span className="sidebar-icon">&#128100;</span>
              {t('menuAdmin')}
            </div>
          )}
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
                      <textarea value={fluxPrompt} onChange={(e) => setFluxPrompt(e.target.value)} rows={8} />
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
                        <label>{t('fluxAspectRatio')}</label>
                        <div className="aspect-ratio-btns">
                          <button className={`ar-btn${fluxAspectRatio === 'portrait' ? ' active' : ''}`} onClick={() => setFluxAspectRatio('portrait')}>{t('fluxPortrait')}</button>
                          <button className={`ar-btn${fluxAspectRatio === 'landscape' ? ' active' : ''}`} onClick={() => setFluxAspectRatio('landscape')}>{t('fluxLandscape')}</button>
                          <button className={`ar-btn${fluxAspectRatio === 'square' ? ' active' : ''}`} onClick={() => setFluxAspectRatio('square')}>{t('fluxSquare')}</button>
                        </div>
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
                    <div className="file-pickers">
                      <button className="btn secondary small" onClick={toggleI2vImagePicker}>
                        {t('selectFromUploads')}
                      </button>
                      <button className="btn secondary small" onClick={toggleI2vOutputPicker}>
                        {t('selectFromOutputs')}
                      </button>
                    </div>
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
                    {showI2vOutputPicker && (
                      <div className="picker-list">
                        {i2vGeneratedImages.length === 0 ? <p className="picker-empty">{t('noOutputs')}</p> : i2vGeneratedImages.map((img) => (
                          <div key={img.filename} className={`picker-item${i2vImagePath === img.path ? ' selected' : ''}`} onClick={() => selectI2vGeneratedImage(img)}>
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

                    <div className="form-group">
                      <label>{t('resolutionLabel')}</label>
                      <div className="resolution-mode-buttons">
                        <button
                          type="button"
                          className={`res-mode-btn ${i2vResolution === '1280*720' ? 'active' : ''}`}
                          onClick={() => setI2vResolution('1280*720')}
                        >
                          {t('i2vPortrait')}
                          <span className="res-mode-size">720 × 1280</span>
                        </button>
                        <button
                          type="button"
                          className={`res-mode-btn ${i2vResolution === '720*1280' ? 'active' : ''}`}
                          onClick={() => setI2vResolution('720*1280')}
                        >
                          {t('i2vLandscape')}
                          <span className="res-mode-size">1280 × 720</span>
                        </button>
                      </div>
                      {i2vImageDimensions && <span className="image-size-info">{t('imageSize')}: {i2vImageDimensions.width} × {i2vImageDimensions.height}</span>}
                    </div>

                    <div className="form-group">
                      <label>{t('i2vFrameNumLabel')}: {i2vFrameNum}</label>
                      <input type="range" min={17} max={121} step={4} value={i2vFrameNum} onChange={(e) => setI2vFrameNum(parseInt(e.target.value))} />
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
                  {loraAdapters.length > 0 && (() => {
                    // Group adapters by type for radio-style selection (one per type)
                    const typeGroups = {};
                    loraAdapters.forEach(a => {
                      const tp = a.type || 'other';
                      if (!typeGroups[tp]) typeGroups[tp] = [];
                      typeGroups[tp].push(a);
                    });
                    const typeOrder = ['character', 'motion', 'camera'];
                    const sortedTypes = Object.keys(typeGroups).sort((a, b) => {
                      const ia = typeOrder.indexOf(a), ib = typeOrder.indexOf(b);
                      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
                    });

                    const handleLoraToggle = (adapter, checked) => {
                      setLoraWeights(prev => ({
                        ...prev,
                        [adapter.name]: { ...prev[adapter.name], enabled: checked },
                      }));
                    };

                    return (
                      <div className="card lora-card">
                        <h3>{t('loraTitle')}</h3>
                        <div className="lora-strategy-info">
                          <p><strong>High-Noise:</strong> {t('loraHighTip')}</p>
                          <p><strong>Low-Noise:</strong> {t('loraLowTip')}</p>
                        </div>
                        {sortedTypes.map(typeName => {
                          const typeLabel = typeName === 'character' ? t('loraCharacter') : typeName === 'camera' ? t('loraCamera') : t('loraMotion');
                          const adaptersInType = typeGroups[typeName];
                          return (
                            <div key={typeName} className="lora-type-group">
                              <div className="lora-type-header">
                                <span className={`lora-type-badge ${typeName}`}>{typeLabel}</span>
                                <span className="lora-type-hint">{adaptersInType.length > 1 ? t('loraMultiSelect') : ''}</span>
                              </div>
                              {adaptersInType.map(adapter => {
                                const w = loraWeights[adapter.name] || { enabled: false, high_weight: 0, low_weight: 0 };
                                const isExpanded = expandedLora === adapter.name;
                                return (
                                  <div key={adapter.name} className={`lora-adapter${w.enabled ? ' enabled' : ''}`}>
                                    <div className="lora-adapter-header">
                                      <label className="lora-toggle">
                                        <input
                                          type="checkbox"
                                          checked={w.enabled}
                                          disabled={!adapter.available}
                                          onChange={(e) => handleLoraToggle(adapter, e.target.checked)}
                                        />
                                        <span className="lora-name">{adapter.name}</span>
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
                                                <span key={tw} className="lora-tag-group">
                                                  <span className="lora-tag">{tw}</span>
                                                  <button
                                                    className="lora-tag-btn lora-tag-add"
                                                    title={t('loraAddToPrompt')}
                                                    onClick={() => {
                                                      if (!i2vPrompt.includes(tw)) {
                                                        setI2vPrompt(prev => prev ? prev + ', ' + tw : tw);
                                                      }
                                                    }}
                                                  >+</button>
                                                  <button
                                                    className="lora-tag-btn lora-tag-copy"
                                                    title={t('loraCopyToClipboard')}
                                                    onClick={(e) => {
                                                      navigator.clipboard.writeText(tw);
                                                      const btn = e.currentTarget;
                                                      btn.textContent = '\u2713';
                                                      setTimeout(() => { btn.textContent = '\u2398'; }, 1500);
                                                    }}
                                                  >{'\u2398'}</button>
                                                </span>
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
                                          <label>
                                            {t('loraHighWeight')}: {w.high_weight.toFixed(2)}
                                            <span className="lora-rec" title={t('loraClickToReset')}
                                              onClick={() => setLoraWeights(prev => ({
                                                ...prev, [adapter.name]: { ...prev[adapter.name], high_weight: adapter.default_high_weight }
                                              }))}>
                                              {t('loraRec')}: {adapter.default_high_weight}
                                            </span>
                                          </label>
                                          <input type="range" min={0} max={1.5} step={0.05} value={w.high_weight}
                                            onChange={(e) => setLoraWeights(prev => ({
                                              ...prev, [adapter.name]: { ...prev[adapter.name], high_weight: parseFloat(e.target.value) }
                                            }))} />
                                        </div>
                                        <div className="form-group">
                                          <label>
                                            {t('loraLowWeight')}: {w.low_weight.toFixed(2)}
                                            <span className="lora-rec" title={t('loraClickToReset')}
                                              onClick={() => setLoraWeights(prev => ({
                                                ...prev, [adapter.name]: { ...prev[adapter.name], low_weight: adapter.default_low_weight }
                                              }))}>
                                              {t('loraRec')}: {adapter.default_low_weight}
                                            </span>
                                          </label>
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
                          );
                        })}
                      </div>
                    );
                  })()}
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

          {/* ============ WORKFLOW ============ */}
          {activeMenu === 'workflow' && workflows.length > 1 && (
            <div className="workflow-tabs">
              {workflows.map(wf => (
                <button key={wf.id}
                  className={`workflow-tab${activeWorkflowId === wf.id ? ' active' : ''}`}
                  onClick={() => { setActiveWorkflowId(wf.id); document.querySelector('.main-content')?.scrollTo(0, 0); }}>
                  {wf.display_name[lang] || wf.display_name.en}
                </button>
              ))}
            </div>
          )}
          {activeMenu === 'workflow' && (
            <div className="page-content">
              {workflows.map(wf => {
                if (wf.id !== activeWorkflowId) return null;
                const wfState = workflowStates[wf.id];
                if (!wfState) return null;
                return (
                  <div key={wf.id}>
                    <h2 className="page-title">{wf.display_name[lang] || wf.display_name.en}</h2>
                    <p className="model-note">{wf.description?.[lang] || wf.description?.en}</p>

                    <div className="two-column">
                      {/* Left: Dynamic Inputs */}
                      <div className="column">
                        {wf.inputs.map(inputDef => renderWorkflowInput(wf.id, inputDef))}

                        {/* YouTube Shorts metadata */}
                        <div className="card yt-meta-card">
                          <h3>{t('ytMeta')}</h3>
                          <div className="yt-meta-fields">
                            <label className="yt-meta-label">{t('ytTitle')}
                              <input type="text" className="yt-meta-input"
                                placeholder={t('ytTitlePlaceholder')}
                                value={wfState.ytTitle || ''}
                                onChange={e => updateWfState(wf.id, { ytTitle: e.target.value })} />
                            </label>
                            <label className="yt-meta-label">{t('ytDescription')}
                              <textarea className="yt-meta-textarea" rows={2}
                                placeholder={t('ytDescPlaceholder')}
                                value={wfState.ytDescription || ''}
                                onChange={e => updateWfState(wf.id, { ytDescription: e.target.value })} />
                            </label>
                            <label className="yt-meta-label">{t('ytHashtags')}
                              <input type="text" className="yt-meta-input"
                                placeholder={t('ytHashPlaceholder')}
                                value={wfState.ytHashtags || ''}
                                onChange={e => updateWfState(wf.id, { ytHashtags: e.target.value })} />
                            </label>
                          </div>
                        </div>

                        <div className="wf-btn-row">
                          {wfState.isGenerating ? (
                            <button
                              className="btn cancel-btn"
                              onClick={() => handleWfCancel(wf.id)}
                              style={{ flex: 1 }}
                            >
                              {t('wfCancelBtn')}
                            </button>
                          ) : (
                            <button
                              className={`btn ${wf.id === 'change_character' ? 'generate-btn-green' : 'generate-btn'}`}
                              onClick={() => handleWfGenerate(wf.id)}
                              disabled={wfQueue[wf.id]?.isProcessing}
                              style={{ flex: 1 }}
                            >
                              {t('wfGenerateBtn')}
                            </button>
                          )}
                          <button
                            className="btn secondary"
                            onClick={() => handleWfQueueAdd(wf.id)}
                            disabled={wfQueue[wf.id]?.isProcessing}
                            style={{ flex: 1 }}
                          >
                            {t('wfAddToQueue')}
                          </button>
                        </div>
                      </div>

                      {/* Right: Output + Queue */}
                      <div className="column">
                        <div className="card">
                          <h3>{t('output')}</h3>
                          {wfState.isGenerating && (
                            <div className="progress-container">
                              <div className="progress-bar">
                                <div className="progress-fill" style={{ width: `${wfState.progress}%` }} />
                              </div>
                              <span className="progress-text">{wfState.progress}%</span>
                            </div>
                          )}
                          {wfState.status && <p className="status-msg">{wfState.status}</p>}
                          {wfState.outputVideo && (
                            <div className="output-container">
                              <video src={wfState.outputVideo} controls style={{ width: '100%', borderRadius: 8 }} />
                              <a href={wfState.outputVideo} download className="btn secondary" style={{ marginTop: 8, display: 'inline-block' }}>
                                {t('download')}
                              </a>
                            </div>
                          )}
                          {!wfState.outputVideo && !wfState.isGenerating && (
                            <p style={{ color: '#888', textAlign: 'center', padding: 40 }}>
                              {t('noOutputYet')}
                            </p>
                          )}
                        </div>

                        {/* Queue Panel */}
                        {(() => {
                          const queue = wfQueue[wf.id];
                          const items = queue?.items || [];
                          const isProcessing = queue?.isProcessing || false;
                          const pendingCount = items.filter(i => i.status === 'pending').length;
                          return (
                            <div className="card queue-card">
                              <h3>{t('wfQueue')} {items.length > 0 && `(${items.length})`}</h3>
                              {items.length === 0 && (
                                <p className="queue-empty">{t('wfQueueEmpty')}</p>
                              )}
                              <div className="queue-list">
                                {items.map(item => (
                                  <div key={item.id} className={`queue-item queue-item--${item.status}`}>
                                    <span className="queue-item-status">
                                      {item.status === 'completed' ? '\u2705' : item.status === 'running' ? '\u23f3' : item.status === 'failed' ? '\u274c' : '\u23f8'}
                                    </span>
                                    <span className="queue-item-label">{item.label}</span>
                                    {item.status === 'running' && (
                                      <div className="queue-item-progress">
                                        <div className="queue-progress-bar">
                                          <div className="queue-progress-fill" style={{ width: `${item.progress}%` }} />
                                        </div>
                                        <span className="queue-progress-text">{item.progress}%</span>
                                      </div>
                                    )}
                                    {item.status === 'pending' && (
                                      <span className="queue-item-pending">{t('wfQueuePending')}</span>
                                    )}
                                    {item.status === 'completed' && item.outputVideo && (
                                      <a href={item.outputVideo} download className="queue-item-dl" title={t('download')}>DL</a>
                                    )}
                                    {item.status === 'failed' && item.error && (
                                      <span className="queue-item-error" title={item.error}>Error</span>
                                    )}
                                    {item.status !== 'running' && (
                                      <button className="queue-item-remove" onClick={() => handleWfQueueRemove(wf.id, item.id)}>&times;</button>
                                    )}
                                  </div>
                                ))}
                              </div>
                              {items.length > 0 && (
                                <div className="queue-actions">
                                  <button className="btn" onClick={() => handleWfQueueStart(wf.id)}
                                    disabled={isProcessing || pendingCount === 0}>
                                    {isProcessing ? t('wfQueueRunning') : t('wfStartQueue')}
                                  </button>
                                  <button className="btn secondary" onClick={() => handleWfQueueClear(wf.id)}
                                    disabled={isProcessing}>
                                    {t('wfClearQueue')}
                                  </button>
                                </div>
                              )}
                            </div>
                          );
                        })()}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* ============ VIDEO STUDIO ============ */}
          {activeMenu === 'studio' && (
            <div className="page-content">
              {/* Mode Tabs */}
              <div className="studio-mode-tabs">
                <button
                  className={`studio-mode-tab${studioMode === 'manual' ? ' active' : ''}`}
                  onClick={() => setStudioMode('manual')}
                >{t('studioManualMode')}</button>
                <button
                  className={`studio-mode-tab${studioMode === 'auto' ? ' active' : ''}`}
                  onClick={() => setStudioMode('auto')}
                >{t('studioAutoMode')}</button>
              </div>

              {studioMode === 'manual' && (
                <>
                  {/* Content Type Selector */}
                  <div className="studio-content-type">
                    <label>{t('studioContentType')}:</label>
                    <div className="studio-content-btns">
                      {['dance', 'narration', 'presentation'].map(ct => (
                        <button key={ct}
                          className={`btn${studioContentType === ct ? ' primary' : ' secondary'}`}
                          onClick={() => studioHandleContentTypeChange(ct)}
                        >
                          {ct === 'dance' ? '🎵 ' : ct === 'narration' ? '🎤 ' : '📊 '}
                          {t(`studioContent${ct.charAt(0).toUpperCase() + ct.slice(1)}`)}
                        </button>
                      ))}
                    </div>
                    <p className="model-note">
                      {studioContentType === 'dance' ? t('studioDanceDesc')
                        : studioContentType === 'narration' ? t('studioNarrationDesc')
                        : t('studioPresentationDesc')}
                    </p>
                  </div>

                  {/* Step Indicator */}
                  <div className="studio-steps">
                    {[1, 2, 3].map(step => (
                      <div key={step} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        {step > 1 && <span className="studio-step-arrow">→</span>}
                        <button
                          className={`studio-step${studioStep === step ? ' active' : ''}${
                            (step === 1 && studioTimeline.length > 0) ||
                            (step === 2 && studioTimeline.length >= 2)
                              ? ' completed' : ''
                          }`}
                          onClick={() => setStudioStep(step)}
                        >
                          {t(`studioStep${step}`)}
                        </button>
                      </div>
                    ))}
                  </div>

                  <div className="two-column">
                    {/* Left Panel - Controls */}
                    <div className="panel">
                      {/* ---- STEP 1: Image Generation ---- */}
                      {studioStep === 1 && (
                        <div className="card">
                          <h3>{t('studioGenerateImage')}</h3>
                          <div className="form-group">
                            <label>{t('fluxPromptLabel')}</label>
                            <textarea
                              value={studioFluxPrompt}
                              onChange={e => setStudioFluxPrompt(e.target.value)}
                              rows={5}
                              placeholder={translations[lang].fluxPromptDefault}
                            />
                          </div>
                          <div className="form-group">
                            <label>{t('studioSeed')}</label>
                            <input type="number" value={studioFluxSeed}
                              onChange={e => setStudioFluxSeed(parseInt(e.target.value) || -1)} />
                          </div>
                          <div className="form-group">
                            <label>{t('fluxAspectRatio')}</label>
                            <div className="aspect-ratio-btns">
                              {['portrait', 'landscape', 'square'].map(ar => (
                                <button key={ar}
                                  className={`btn${studioFluxAspectRatio === ar ? ' primary' : ' secondary'}`}
                                  onClick={() => setStudioFluxAspectRatio(ar)}
                                >{t(`flux${ar.charAt(0).toUpperCase() + ar.slice(1)}`)}</button>
                              ))}
                            </div>
                          </div>

                          {/* LoRA Selection for Image Gen */}
                          {imgLoraAdapters.length > 0 && (
                            <div className="form-group">
                              <label>{t('loraTitle')}</label>
                              <div className="lora-list">
                                {imgLoraAdapters.filter(a => a.available).map(a => (
                                  <div key={a.name} className="lora-item">
                                    <label className="lora-toggle">
                                      <input type="checkbox"
                                        checked={studioSelectedLoras[a.name]?.enabled || false}
                                        onChange={e => setStudioSelectedLoras(prev => ({
                                          ...prev,
                                          [a.name]: { ...prev[a.name], enabled: e.target.checked,
                                            high_weight: prev[a.name]?.high_weight ?? a.default_high_weight,
                                            low_weight: prev[a.name]?.low_weight ?? a.default_low_weight },
                                        }))}
                                      />
                                      <span>{a.name}</span>
                                    </label>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          <button className="btn primary"
                            disabled={studioFluxIsGenerating || !studioFluxPrompt.trim()}
                            onClick={studioHandleFluxGenerate}
                          >
                            {studioFluxIsGenerating
                              ? `${t('fluxGenerating')} ${studioFluxProgress}%`
                              : t('studioGenerateImage')}
                          </button>

                          <div style={{ marginTop: 12 }}>
                            <button className="btn secondary" onClick={() => {
                              studioLoadGallery();
                              setStudioGalleryOpen(true);
                            }}>{t('studioSelectFromGallery')}</button>
                          </div>
                        </div>
                      )}

                      {/* ---- STEP 2: Timeline Controls ---- */}
                      {studioStep === 2 && (
                        <div className="card">
                          <h3>{t('studioStep2')}</h3>
                          <p className="model-note">{t('studioDragToReorder')}</p>
                          {studioTimeline.length === 0 ? (
                            <p className="model-note">{t('studioMinImages')}</p>
                          ) : (
                            <div>
                              <p>{t('studioTotalFrames')}: {studioTimeline.reduce((s, i) => s + i.segmentLength, 0)}</p>
                              <p>{t('studioEstDuration')}: {(studioTimeline.reduce((s, i) => s + i.segmentLength, 0) / 24).toFixed(1)}s</p>
                            </div>
                          )}
                          <button className="btn secondary" style={{ marginTop: 8 }} onClick={() => {
                            studioLoadGallery();
                            setStudioGalleryOpen(true);
                          }}>{t('studioSelectFromGallery')}</button>
                        </div>
                      )}

                      {/* ---- STEP 3: Video Generation ---- */}
                      {studioStep === 3 && (
                        <div className="card">
                          <h3>{t('studioStep3')}</h3>

                          {/* Video Mode Tabs */}
                          <div className="sub-tabs" style={{ marginBottom: 16 }}>
                            <button className={studioVideoMode === 'fflf' ? 'active' : ''}
                              onClick={() => setStudioVideoMode('fflf')}>{t('studioFflfMode')}</button>
                            <button className={studioVideoMode === 'infinitalk' ? 'active' : ''}
                              onClick={() => setStudioVideoMode('infinitalk')}>{t('studioInfiniTalkMode')}</button>
                            <button className={studioVideoMode === 'change_character' ? 'active' : ''}
                              onClick={() => setStudioVideoMode('change_character')}>{t('studioChangeCharMode')}</button>
                          </div>

                          {/* FFLF Mode */}
                          {studioVideoMode === 'fflf' && (
                            <>
                              <div className="form-group">
                                <label>{t('studioMasterPrompt')}</label>
                                <textarea value={studioMasterPrompt} onChange={e => setStudioMasterPrompt(e.target.value)} rows={3} />
                              </div>
                              <div className="form-group">
                                <label>{t('studioNegPrompt')}</label>
                                <textarea value={studioNegPrompt} onChange={e => setStudioNegPrompt(e.target.value)} rows={2} />
                              </div>
                              <div className="form-group">
                                <label>{t('studioLooping')}</label>
                                <label className="lora-toggle">
                                  <input type="checkbox" checked={studioLooping} onChange={e => setStudioLooping(e.target.checked)} />
                                  <span>{studioLooping ? 'ON' : 'OFF'}</span>
                                </label>
                              </div>
                              <div className="form-group">
                                <label>{t('studioInitialWidth')}: {studioInitialWidth}</label>
                                <input type="range" min={192} max={480} step={16} value={studioInitialWidth}
                                  onChange={e => setStudioInitialWidth(parseInt(e.target.value))} />
                              </div>
                              <div className="form-group">
                                <label>{t('studioUpscaleFactor')}: {studioUpscaleFactor}</label>
                                <input type="range" min={1} max={4} step={0.5} value={studioUpscaleFactor}
                                  onChange={e => setStudioUpscaleFactor(parseFloat(e.target.value))} />
                              </div>
                              <div className="form-group">
                                <label>{t('studioVideoSeed')}</label>
                                <input type="number" value={studioVideoSeed} onChange={e => setStudioVideoSeed(parseInt(e.target.value) || 0)} />
                              </div>
                              <button className="btn primary" disabled={studioIsGenerating || studioTimeline.length < 2}
                                onClick={studioHandleCreateVideo}>
                                {studioIsGenerating ? `${t('studioCreateVideo')} ${studioProgress}%` : t('studioCreateVideo')}
                              </button>
                            </>
                          )}

                          {/* InfiniTalk Mode */}
                          {studioVideoMode === 'infinitalk' && (
                            <>
                              <p className="model-note">{t('studioNarrationDesc')}</p>
                              <div className="form-group">
                                <label>{t('studioMasterPrompt')}</label>
                                <textarea value={studioMasterPrompt} onChange={e => setStudioMasterPrompt(e.target.value)} rows={2} />
                              </div>

                              {/* Audio Source Toggle */}
                              <div className="sub-tabs" style={{ marginBottom: 12 }}>
                                <button className={studioAudioSource === 'tts' ? 'active' : ''}
                                  onClick={() => setStudioAudioSource('tts')}>{t('studioAudioSourceTts')}</button>
                                <button className={studioAudioSource === 'upload' ? 'active' : ''}
                                  onClick={() => setStudioAudioSource('upload')}>{t('studioAudioSourceUpload')}</button>
                              </div>

                              {studioAudioSource === 'tts' && (
                                <>
                                  <div className="form-group">
                                    <label>{t('studioTtsScript')}</label>
                                    <textarea value={studioTtsScript} onChange={e => setStudioTtsScript(e.target.value)} rows={4}
                                      placeholder="Enter narration text..." />
                                  </div>
                                  <div className="form-group" style={{ display: 'flex', gap: 8 }}>
                                    <div style={{ flex: 1 }}>
                                      <label>{t('studioTtsLanguage')}</label>
                                      <select value={studioTtsLanguage} onChange={e => setStudioTtsLanguage(e.target.value)}>
                                        {['Korean', 'English', 'Chinese', 'Japanese'].map(l => (
                                          <option key={l} value={l}>{l}</option>
                                        ))}
                                      </select>
                                    </div>
                                    <div style={{ flex: 1 }}>
                                      <label>{t('studioTtsSpeaker')}</label>
                                      <select value={studioTtsSpeaker} onChange={e => setStudioTtsSpeaker(e.target.value)}>
                                        {['Ryan', 'Claire', 'Laura', 'Aidan', 'Matt', 'Aria', 'Serena', 'Leo', 'Mei', 'Luna'].map(s => (
                                          <option key={s} value={s}>{s}</option>
                                        ))}
                                      </select>
                                    </div>
                                  </div>
                                  <button className="btn secondary" disabled={studioTtsGenerating || !studioTtsScript.trim()}
                                    onClick={studioHandleTTSGenerate}>
                                    {studioTtsGenerating ? '...' : t('studioTtsGenerate')}
                                  </button>
                                </>
                              )}

                              {studioAudioSource === 'upload' && (
                                <div className="form-group">
                                  <label>{t('studioUploadAudio')}</label>
                                  <div className="upload-area">
                                    <input type="file" accept="audio/*" onChange={studioHandleAudioUpload} />
                                    <p>{t('dropAudio')}</p>
                                  </div>
                                </div>
                              )}

                              {/* Audio Preview */}
                              {studioNarrationAudio && (
                                <div style={{ margin: '12px 0' }}>
                                  <audio controls src={studioNarrationAudio} style={{ width: '100%' }} />
                                  <p className="model-note">Frames: {studioNarrationLength} (~{(studioNarrationLength / 20).toFixed(1)}s at 20fps)</p>
                                </div>
                              )}

                              <button className="btn primary" disabled={studioIsGenerating || !studioNarrationAudioPath || studioTimeline.length < 1}
                                onClick={studioHandleInfiniTalk}>
                                {studioIsGenerating ? `Generating... ${studioProgress}%` : t('studioCreateVideo')}
                              </button>
                            </>
                          )}

                          {/* Change Character Mode */}
                          {studioVideoMode === 'change_character' && (
                            <>
                              <div className="form-group">
                                <label>{t('studioSceneDesc')}</label>
                                <textarea value={studioScenePrompt} onChange={e => setStudioScenePrompt(e.target.value)} rows={2} />
                              </div>

                              {/* Ref Video Source */}
                              <div className="sub-tabs" style={{ marginBottom: 12 }}>
                                <button className={studioRefVideoMode === 'upload' ? 'active' : ''}
                                  onClick={() => setStudioRefVideoMode('upload')}>{t('wfVideoUpload')}</button>
                                <button className={studioRefVideoMode === 'youtube' ? 'active' : ''}
                                  onClick={() => setStudioRefVideoMode('youtube')}>{t('wfVideoYoutube')}</button>
                              </div>

                              {studioRefVideoMode === 'upload' && (
                                <div className="form-group">
                                  <label>{t('studioRefVideo')}</label>
                                  <div className="upload-area">
                                    <input type="file" accept="video/*" onChange={studioHandleRefVideoUpload} />
                                    <p>{t('dropVideo')}</p>
                                  </div>
                                </div>
                              )}

                              {studioRefVideoMode === 'youtube' && (
                                <div className="form-group youtube-input">
                                  <input type="text" placeholder="YouTube URL"
                                    value={studioYoutubeUrl} onChange={e => setStudioYoutubeUrl(e.target.value)} />
                                  <button className="btn secondary" disabled={studioYoutubeDownloading}
                                    onClick={studioHandleYoutubeDownload}>
                                    {studioYoutubeDownloading ? t('wfDownloading') : t('wfDownloadBtn')}
                                  </button>
                                </div>
                              )}

                              {studioRefVideoPreview && (
                                <video controls src={studioRefVideoPreview} style={{ width: '100%', marginTop: 8, borderRadius: 8 }} />
                              )}

                              <div className="form-group">
                                <label>{t('studioAspectRatio')}</label>
                                <div className="aspect-ratio-btns">
                                  {['portrait', 'landscape', 'square'].map(ar => (
                                    <button key={ar}
                                      className={`btn${studioFluxAspectRatio === ar ? ' primary' : ' secondary'}`}
                                      onClick={() => setStudioFluxAspectRatio(ar)}
                                    >{t(`flux${ar.charAt(0).toUpperCase() + ar.slice(1)}`)}</button>
                                  ))}
                                </div>
                              </div>

                              <button className="btn primary" disabled={studioIsGenerating || studioTimeline.length < 1 || !studioRefVideoPath}
                                onClick={studioHandleChangeCharacter}>
                                {studioIsGenerating ? `Generating... ${studioProgress}%` : t('studioCreateVideo')}
                              </button>
                            </>
                          )}

                          {/* Status & Output */}
                          {studioStatus && (
                            <div className="status-box" style={{ marginTop: 16 }}>
                              <p>{studioStatus}</p>
                              {studioIsGenerating && (
                                <div className="progress-bar">
                                  <div className="progress-fill" style={{ width: `${studioProgress}%` }} />
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Right Panel - Preview */}
                    <div className="panel">
                      {/* Timeline (always visible in right panel) */}
                      <div className="card">
                        <h3>{t('studioStep2')}</h3>
                        <div className="studio-timeline">
                          {studioTimeline.length === 0 ? (
                            <p className="model-note" style={{ margin: 'auto' }}>{t('studioMinImages')}</p>
                          ) : (
                            studioTimeline.map((item, idx) => (
                              <div key={item.id} style={{ display: 'flex', alignItems: 'flex-start' }}>
                                {idx > 0 && (
                                  <span className="studio-timeline-connector">→</span>
                                )}
                                <div
                                  className={`studio-timeline-item${studioDragIndex === idx ? ' dragging' : ''}`}
                                  draggable
                                  onDragStart={e => studioHandleDragStart(e, idx)}
                                  onDragOver={e => studioHandleDragOver(e, idx)}
                                  onDragEnd={studioHandleDragEnd}
                                >
                                  <button className="studio-timeline-remove"
                                    onClick={() => studioRemoveFromTimeline(idx)}>×</button>
                                  <img src={item.imageUrl} alt={`Frame ${idx + 1}`} />
                                  <input type="number" className="studio-segment-input"
                                    value={item.segmentLength}
                                    onChange={e => studioUpdateSegmentLength(idx, parseInt(e.target.value) || 33)}
                                    min={17} max={81} />
                                  <span style={{ fontSize: '0.7rem', color: 'var(--text-light)' }}>{t('studioSegmentFrames')}</span>
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </div>

                      {/* Video Output */}
                      {studioOutputVideo && (
                        <div className="card" style={{ marginTop: 16 }}>
                          <h3>{t('output')}</h3>
                          <video controls src={studioOutputVideo} style={{ width: '100%', borderRadius: 8 }} />
                          <a href={studioOutputVideo} download className="btn secondary" style={{ marginTop: 8, display: 'block', textAlign: 'center' }}>
                            {t('download')}
                          </a>
                        </div>
                      )}
                    </div>
                  </div>
                </>
              )}

              {/* Auto Mode (AI Chat) */}
              {studioMode === 'auto' && (
                <div className="two-column">
                  {/* Chat Panel */}
                  <div className="panel">
                    <div className="card studio-chat">
                      <div className="studio-chat-messages" ref={el => { if (el) el.scrollTop = el.scrollHeight; }}>
                        {chatMessages.map((msg, idx) => (
                          <div key={idx}>
                            <div className={`studio-chat-bubble ${msg.role}`}>
                              {msg.text}
                            </div>
                            {msg.actions && msg.actions.length > 0 && (
                              <div className="studio-chat-actions">
                                {msg.actions.map((action, ai) => (
                                  action.result?.output_path && (
                                    <img key={ai} src={action.result.output_path}
                                      className="studio-chat-action-image"
                                      alt="Generated"
                                      onClick={() => studioAddToTimeline(action.result.output_path, action.result.absolute_path || action.result.output_path)}
                                    />
                                  )
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                        {chatLoading && (
                          <div className="studio-chat-typing">
                            <span /><span /><span />
                          </div>
                        )}
                      </div>
                      <div className="studio-chat-input-area">
                        <input
                          className="studio-chat-input"
                          value={chatInput}
                          onChange={e => setChatInput(e.target.value)}
                          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && studioHandleChatSend()}
                          placeholder="Ask AI to create a video..."
                        />
                        <button className="studio-chat-send" disabled={chatLoading || !chatInput.trim()}
                          onClick={studioHandleChatSend}>▶</button>
                      </div>
                    </div>
                  </div>

                  {/* Right Panel - Timeline & Output (reused from manual mode) */}
                  <div className="panel">
                    <div className="card">
                      <h3>{t('studioStep2')}</h3>
                      <div className="studio-timeline">
                        {studioTimeline.length === 0 ? (
                          <p className="model-note" style={{ margin: 'auto' }}>{t('studioMinImages')}</p>
                        ) : (
                          studioTimeline.map((item, idx) => (
                            <div key={item.id} style={{ display: 'flex', alignItems: 'flex-start' }}>
                              {idx > 0 && <span className="studio-timeline-connector">→</span>}
                              <div className={`studio-timeline-item${studioDragIndex === idx ? ' dragging' : ''}`}
                                draggable onDragStart={e => studioHandleDragStart(e, idx)}
                                onDragOver={e => studioHandleDragOver(e, idx)} onDragEnd={studioHandleDragEnd}>
                                <button className="studio-timeline-remove" onClick={() => studioRemoveFromTimeline(idx)}>×</button>
                                <img src={item.imageUrl} alt={`Frame ${idx + 1}`} />
                                <input type="number" className="studio-segment-input" value={item.segmentLength}
                                  onChange={e => studioUpdateSegmentLength(idx, parseInt(e.target.value) || 33)} min={17} max={81} />
                              </div>
                            </div>
                          ))
                        )}
                      </div>
                    </div>
                    {studioOutputVideo && (
                      <div className="card" style={{ marginTop: 16 }}>
                        <h3>{t('output')}</h3>
                        <video controls src={studioOutputVideo} style={{ width: '100%', borderRadius: 8 }} />
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Gallery Picker Modal */}
              {studioGalleryOpen && (
                <div className="modal-overlay" onClick={() => setStudioGalleryOpen(false)}>
                  <div className="modal-content" onClick={e => e.stopPropagation()}>
                    <h3>{t('studioSelectFromGallery')}</h3>
                    <div className="gallery-grid">
                      {studioGalleryImages.map(img => (
                        <div key={img.filename} className="gallery-item" style={{ cursor: 'pointer' }}
                          onClick={() => {
                            studioAddToTimeline(img.url, img.path || img.url);
                            setStudioGalleryOpen(false);
                          }}>
                          <img src={img.url} alt={img.filename} className="gallery-item-img" />
                          <span className="gallery-item-name">{img.filename}</span>
                        </div>
                      ))}
                    </div>
                    <button className="btn secondary" style={{ marginTop: 12 }}
                      onClick={() => setStudioGalleryOpen(false)}>Close</button>
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

          {/* ============ ADMIN PANEL ============ */}
          {activeMenu === 'admin' && authUser.role === 'superadmin' && (
            <div className="page-content">
              <h2 className="page-title">{t('adminTitle')}</h2>
              <div className="card">
                <div className="admin-table-wrapper">
                  {adminLoading ? (
                    <div style={{ textAlign: 'center', padding: '2rem' }}><div className="loading-spinner" style={{ margin: '0 auto' }} /></div>
                  ) : (
                    <table className="admin-table">
                      <thead>
                        <tr>
                          <th>{t('adminEmail')}</th>
                          <th>{t('adminName')}</th>
                          <th>{t('adminRole')}</th>
                          <th>{t('adminStatus')}</th>
                          <th>{t('adminCreated')}</th>
                          <th>{t('adminLastLogin')}</th>
                          <th>{t('adminActions')}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {adminUsers.map(u => (
                          <tr key={u.id}>
                            <td>
                              <div className="admin-user-info">
                                {u.picture && <img src={u.picture} alt="" className="admin-user-avatar" referrerPolicy="no-referrer" />}
                                {u.email}
                              </div>
                            </td>
                            <td>{u.name || '-'}</td>
                            <td><span className={`badge badge-${u.role}`}>{u.role}</span></td>
                            <td><span className={`badge badge-${u.status}`}>{t(`admin${u.status.charAt(0).toUpperCase() + u.status.slice(1)}`)}</span></td>
                            <td>{u.created_at ? new Date(u.created_at).toLocaleDateString() : '-'}</td>
                            <td>{u.last_login ? new Date(u.last_login).toLocaleString() : '-'}</td>
                            <td>
                              <div className="admin-actions">
                                {u.status === 'pending' && (
                                  <button className="btn success small" onClick={() => handleAdminAction(u.id, 'approve')}>{t('adminApprove')}</button>
                                )}
                                {u.status === 'approved' && u.role !== 'superadmin' && (
                                  <button className="btn warning small" onClick={() => handleAdminAction(u.id, 'suspend')}>{t('adminSuspend')}</button>
                                )}
                                {u.status === 'suspended' && (
                                  <button className="btn success small" onClick={() => handleAdminAction(u.id, 'activate')}>{t('adminActivate')}</button>
                                )}
                                {u.role !== 'superadmin' && (
                                  <button className="btn danger small" onClick={() => handleAdminAction(u.id, 'delete')}>{t('adminDelete')}</button>
                                )}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
