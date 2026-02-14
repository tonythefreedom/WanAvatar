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
  uploadToGallery,
  listBackgrounds,
  listAvatarGroups,
  listAvatarImages,
  deleteAvatarImage,
  registerAvatar,
  prepareAvatar,
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
  uploadToYouTube,
  deleteBackground,
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
    galleryTitle: 'Assets',
    galleryEmpty: 'No videos generated yet',
    galleryDelete: 'Delete',
    galleryDeleteConfirm: 'Delete this video?',
    gallerySize: 'Size',
    galleryDate: 'Created',
    galleryRefresh: 'Refresh',
    galleryUploadBtn: 'Upload',
    galleryUploading: 'Uploading',
    galleryYtUpload: 'YouTube Upload',
    galleryYtTitle: 'Title',
    galleryYtDesc: 'Description',
    galleryYtHashtags: 'Hashtags',
    galleryYtUploading: 'Uploading...',
    galleryYtSuccess: 'Uploaded!',
    galleryYtCancel: 'Cancel',
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
    // Dance Shorts
    menuDanceShorts: 'Dance Shorts',
    dsCharImage: 'Character Image',
    dsSelectAvatar: 'Select Avatar',
    dsNeedImage: 'Please select or upload a character image',
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
    galleryStages: 'Stages',
    // Output picker
    selectFromOutputs: 'Select from generated outputs',
    noOutputs: 'No generated outputs available',
    outputTypeImage: 'Image',
    outputTypeVideo: 'Video (first frame)',
    // Workflow (generic)
    menuWorkflow: 'Video Studio',
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
    wfClearVideo: 'Clear',
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
    studioChangeCharMode: 'Dance Shorts',
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
    studioAvatarName: 'Avatar Name',
    studioStageSelect: 'Select Stage',
    studioStageUpload: 'Upload Stage',
    studioBgEffect: 'Background Effect',
    studioStageDeleteConfirm: 'Delete this stage?',
    studioYtChannel: 'YouTube Channel URL',
    studioYtChannelName: 'Channel Name',
    studioYtSettings: 'YouTube Shorts Settings',
    studioYtUploadBtn: 'Upload to YouTube',
    studioYtUploading: 'Uploading to YouTube...',
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
    galleryTitle: '에셋',
    galleryEmpty: '아직 생성된 비디오가 없습니다',
    galleryDelete: '삭제',
    galleryDeleteConfirm: '이 비디오를 삭제하시겠습니까?',
    gallerySize: '크기',
    galleryDate: '생성일',
    galleryRefresh: '새로고침',
    galleryUploadBtn: '업로드',
    galleryUploading: '업로드 중',
    galleryYtUpload: 'YouTube 업로드',
    galleryYtTitle: '제목',
    galleryYtDesc: '설명',
    galleryYtHashtags: '해시태그',
    galleryYtUploading: '업로드 중...',
    galleryYtSuccess: '업로드 완료!',
    galleryYtCancel: '취소',
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
    // Dance Shorts
    menuDanceShorts: '댄스 쇼츠',
    dsCharImage: '캐릭터 이미지',
    dsSelectAvatar: '아바타 선택',
    dsNeedImage: '캐릭터 이미지를 선택하거나 업로드해 주세요',
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
    galleryStages: '스테이지',
    // Output picker
    selectFromOutputs: '생성된 결과에서 선택',
    noOutputs: '생성된 결과가 없습니다',
    outputTypeImage: '이미지',
    outputTypeVideo: '비디오 (첫 프레임)',
    // Workflow (generic)
    menuWorkflow: '비디오 스튜디오',
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
    wfClearVideo: '초기화',
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
    studioChangeCharMode: '댄스 쇼츠',
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
    studioAvatarName: '아바타 이름',
    studioStageSelect: '스테이지 선택',
    studioStageUpload: '스테이지 업로드',
    studioBgEffect: '배경 효과',
    studioStageDeleteConfirm: '이 스테이지를 삭제하시겠습니까?',
    studioYtChannel: 'YouTube 채널 URL',
    studioYtChannelName: '채널명',
    studioYtSettings: 'YouTube Shorts 설정',
    studioYtUploadBtn: 'YouTube 업로드',
    studioYtUploading: 'YouTube 업로드 중...',
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
    galleryTitle: '资源',
    galleryEmpty: '尚未生成任何视频',
    galleryDelete: '删除',
    galleryDeleteConfirm: '确定删除此视频？',
    gallerySize: '大小',
    galleryDate: '创建时间',
    galleryRefresh: '刷新',
    galleryUploadBtn: '上传',
    galleryUploading: '上传中',
    galleryYtUpload: 'YouTube上传',
    galleryYtTitle: '标题',
    galleryYtDesc: '描述',
    galleryYtHashtags: '标签',
    galleryYtUploading: '上传中...',
    galleryYtSuccess: '上传成功!',
    galleryYtCancel: '取消',
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
    // Dance Shorts
    menuDanceShorts: '舞蹈短片',
    dsCharImage: '角色图片',
    dsSelectAvatar: '选择角色',
    dsNeedImage: '请选择或上传角色图片',
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
    galleryStages: '舞台',
    // Output picker
    selectFromOutputs: '从生成结果中选择',
    noOutputs: '没有生成结果',
    outputTypeImage: '图片',
    outputTypeVideo: '视频（首帧）',
    // Workflow
    menuWorkflow: '视频工作室',
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
    wfClearVideo: '清除',
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
    studioChangeCharMode: '舞蹈短片',
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
    studioAvatarName: '角色名称',
    studioStageSelect: '选择舞台',
    studioStageUpload: '上传舞台',
    studioBgEffect: '背景效果',
    studioStageDeleteConfirm: '确定删除此舞台？',
    studioYtChannel: 'YouTube频道链接',
    studioYtChannelName: '频道名称',
    studioYtSettings: 'YouTube Shorts设置',
    studioYtUploadBtn: '上传到YouTube',
    studioYtUploading: 'YouTube上传中...',
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

// ─── localStorage helpers for persisting queue state across refresh ───
const WF_QUEUE_KEY = 'cinesynth_wf_queue';
const saveQueueToStorage = (queue) => {
  try { localStorage.setItem(WF_QUEUE_KEY, JSON.stringify(queue)); } catch {}
};
const loadQueueFromStorage = () => {
  try { return JSON.parse(localStorage.getItem(WF_QUEUE_KEY) || '{}'); } catch { return {}; }
};

// ─── localStorage helpers for persisting active generation tasks across refresh ───
const ACTIVE_TASKS_KEY = 'cinesynth_active_tasks';
const saveActiveTask = (wfId, taskId) => {
  try {
    const tasks = JSON.parse(localStorage.getItem(ACTIVE_TASKS_KEY) || '{}');
    tasks[wfId] = taskId;
    localStorage.setItem(ACTIVE_TASKS_KEY, JSON.stringify(tasks));
  } catch {}
};
const removeActiveTask = (wfId) => {
  try {
    const tasks = JSON.parse(localStorage.getItem(ACTIVE_TASKS_KEY) || '{}');
    delete tasks[wfId];
    localStorage.setItem(ACTIVE_TASKS_KEY, JSON.stringify(tasks));
  } catch {}
};
const getActiveTasks = () => {
  try { return JSON.parse(localStorage.getItem(ACTIVE_TASKS_KEY) || '{}'); }
  catch { return {}; }
};

function App() {
  const [lang, setLang] = useState('en');
  const [activeMenu, setActiveMenu] = useState('danceshorts');
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
  const [galleryUploading, setGalleryUploading] = useState(false);
  const [galleryUploadProgress, setGalleryUploadProgress] = useState(0);
  const galleryUploadRef = useRef(null);

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
  const [avatarPreparing, setAvatarPreparing] = useState({}); // { [filename]: { taskId, status, progress } }
  const [ytUploadTarget, setYtUploadTarget] = useState(null); // filename being uploaded
  const [ytUploadForm, setYtUploadForm] = useState({ title: '', description: '', hashtags: '' });
  const [ytUploadStatus, setYtUploadStatus] = useState(null); // null | 'uploading' | 'success' | 'error'
  const [ytUploadResult, setYtUploadResult] = useState(null); // youtube URL or error message

  // Avatar image popup viewer
  const [avatarPopup, setAvatarPopup] = useState(null); // { url, filename, group, wfId, img }

  // Image-category LoRA state (separate from mov LoRAs)
  const [imgLoraAdapters, setImgLoraAdapters] = useState([]);
  const [imgLoraWeights, setImgLoraWeights] = useState({});
  const [expandedImgLora, setExpandedImgLora] = useState(null);

  // Workflow state (dynamic, per-workflow)
  const [workflows, setWorkflows] = useState([]);
  const [activeWorkflowId, setActiveWorkflowId] = useState(null);
  const [workflowStates, setWorkflowStates] = useState({});
  const [wfQueue, setWfQueue] = useState(() => {
    const saved = loadQueueFromStorage();
    // Restore: keep running items with taskId alive, revert others to pending
    for (const wfId of Object.keys(saved)) {
      if (!saved[wfId]) { delete saved[wfId]; continue; }
      if (saved[wfId].items) {
        const hasRunningWithTask = saved[wfId].items.some(i => i.status === 'running' && i.taskId);
        saved[wfId].isProcessing = hasRunningWithTask;
        saved[wfId].items = saved[wfId].items
          .filter(i => i.status !== 'completed' && i.status !== 'failed')
          .map(i => {
            if (i.status === 'running' && i.taskId) return i;
            if (i.status === 'running') return { ...i, status: 'pending', progress: 0 };
            return i;
          });
        if (saved[wfId].items.length === 0) delete saved[wfId];
      } else {
        saved[wfId].isProcessing = false;
      }
    }
    return saved;
  });
  const wfQueueRef = useRef({});
  const videoRefsMap = useRef({});
  const timelineRefsMap = useRef({});
  const tasksRestoredRef = useRef(false);
  const queueRestoredRef = useRef(false);

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

  // Dance Shorts (Change Character)
  const [studioRefVideoPath, setStudioRefVideoPath] = useState('');
  const [studioRefVideoPreview, setStudioRefVideoPreview] = useState(null);
  const [studioRefVideoMode, setStudioRefVideoMode] = useState('youtube');
  const [studioYoutubeUrl, setStudioYoutubeUrl] = useState('');
  const [studioYoutubeDownloading, setStudioYoutubeDownloading] = useState(false);
  const [studioScenePrompt, setStudioScenePrompt] = useState('The character is dancing in the room');

  // Dance Shorts - Video Timeline / Trim
  const [dsVideoDuration, setDsVideoDuration] = useState(0);
  const [dsTrimStart, setDsTrimStart] = useState(0);
  const [dsTrimEnd, setDsTrimEnd] = useState(0);
  const [dsTrimming, setDsTrimming] = useState(false);
  const [dsPlayheadPosition, setDsPlayheadPosition] = useState(0);
  const dsVideoRef = useRef(null);
  const dsTimelineRef = useRef(null);

  // Dance Shorts - Character Image (direct upload or avatar gallery)
  const [dsCharImagePath, setDsCharImagePath] = useState('');
  const [dsCharImagePreview, setDsCharImagePreview] = useState(null);
  const [dsAvatarGroups, setDsAvatarGroups] = useState([]);
  const [dsAvatarSelectedGroup, setDsAvatarSelectedGroup] = useState('');
  const [dsAvatarImages, setDsAvatarImages] = useState([]);

  // Dance Shorts - Avatar & Stage
  const [studioAvatarName, setStudioAvatarName] = useState('');
  const [studioStages, setStudioStages] = useState([]);
  const [studioSelectedStage, setStudioSelectedStage] = useState(null);
  const [studioBgPrompt, setStudioBgPrompt] = useState('light color and strength keep changing');

  // Dance Shorts - YouTube Settings
  const DEFAULT_YT_CHANNEL = 'https://www.youtube.com/channel/UCYcITGLPC3qv9txSM4GB70w';
  const [studioYtChannel, setStudioYtChannel] = useState(DEFAULT_YT_CHANNEL);
  const [studioYtChannelName, setStudioYtChannelName] = useState('');
  const [studioYtTitle, setStudioYtTitle] = useState('');
  const [studioYtDescription, setStudioYtDescription] = useState('');
  const [studioYtHashtags, setStudioYtHashtags] = useState('');
  const [studioYtUploadStatus, setStudioYtUploadStatus] = useState(null);
  const [studioYtUploadResult, setStudioYtUploadResult] = useState(null);
  const [studioCurrentTaskId, setStudioCurrentTaskId] = useState(null);

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
  // Load config (public endpoint — needed for google_client_id on login page)
  useEffect(() => {
    getConfig().then(res => setConfig(res.data)).catch(console.error);
  }, []);

  // Load app data after authentication
  useEffect(() => {
    if (!authUser) return;

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
  }, [authUser]);

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

  // ─── Restore active generation tasks after page refresh ───
  useEffect(() => {
    if (tasksRestoredRef.current || Object.keys(workflowStates).length === 0) return;
    tasksRestoredRef.current = true;
    const activeTasks = getActiveTasks();
    Object.entries(activeTasks).forEach(([wfId, taskId]) => {
      if (!workflowStates[wfId]) { removeActiveTask(wfId); return; }
      updateWfState(wfId, { isGenerating: true, currentTaskId: taskId, status: t('wfGenerating') });
      const poll = setInterval(async () => {
        try {
          const s = await getTaskStatus(taskId);
          updateWfState(wfId, {
            progress: Math.round((s.progress || 0) * 100),
            status: s.status_message || s.status,
          });
          if (s.status === 'completed') {
            clearInterval(poll);
            removeActiveTask(wfId);
            updateWfState(wfId, { isGenerating: false, progress: 100, outputVideo: s.output_url, currentTaskId: null });
          } else if (s.status === 'failed') {
            clearInterval(poll);
            removeActiveTask(wfId);
            updateWfState(wfId, { isGenerating: false, status: `Error: ${s.error || 'Failed'}`, currentTaskId: null });
          } else if (s.status === 'cancelled') {
            clearInterval(poll);
            removeActiveTask(wfId);
            updateWfState(wfId, { isGenerating: false, status: t('wfCancelled'), currentTaskId: null });
          }
        } catch (err) {
          clearInterval(poll);
          removeActiveTask(wfId);
          updateWfState(wfId, { isGenerating: false, status: `Polling error: ${err.message}`, currentTaskId: null });
        }
      }, 3000);
    });
  }, [workflowStates, updateWfState, t]); // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Restore queue processing tasks after page refresh ───
  useEffect(() => {
    if (queueRestoredRef.current) return;
    queueRestoredRef.current = true;
    for (const [wfId, q] of Object.entries(wfQueue)) {
      if (!q?.items) continue;
      const runningItem = q.items.find(i => i.status === 'running' && i.taskId);
      if (!runningItem) continue;
      const { id: itemId, taskId } = runningItem;
      // Resume polling for the running queue item
      const poll = setInterval(async () => {
        try {
          const s = await getTaskStatus(taskId);
          updateQueueItem(wfId, itemId, { progress: Math.round((s.progress || 0) * 100) });
          if (s.status === 'completed') {
            clearInterval(poll);
            updateQueueItem(wfId, itemId, { status: 'completed', progress: 100, outputVideo: s.output_url || s.output_path, taskId: null });
            // Continue processing remaining pending items
            setTimeout(() => handleWfQueueStart(wfId), 500);
          } else if (s.status === 'failed') {
            clearInterval(poll);
            updateQueueItem(wfId, itemId, { status: 'failed', error: s.message || 'Failed', taskId: null });
            setTimeout(() => handleWfQueueStart(wfId), 500);
          } else if (s.status === 'cancelled') {
            clearInterval(poll);
            updateQueueItem(wfId, itemId, { status: 'failed', error: 'Cancelled', taskId: null });
            setTimeout(() => handleWfQueueStart(wfId), 500);
          }
        } catch (err) {
          clearInterval(poll);
          updateQueueItem(wfId, itemId, { status: 'failed', error: err.message, taskId: null });
          setTimeout(() => handleWfQueueStart(wfId), 500);
        }
      }, 3000);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
      id: Math.random().toString(36).slice(2) + Date.now().toString(36), imageUrl, imagePath,
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

  // Dance Shorts generation — add to queue and auto-start
  const studioHandleChangeCharacter = () => {
    if (!dsCharImagePath) return alert(t('dsNeedImage'));
    if (!studioRefVideoPath) return alert(t('studioNeedRefVideo'));
    const defaults = getStudioYtDefaults();
    const ytTitle = studioYtTitle.trim() || defaults.title;
    const ytDesc = studioYtDescription.trim() || defaults.description;
    const ytHash = studioYtHashtags.trim() || defaults.hashtags;
    const item = {
      id: Math.random().toString(36).slice(2) + Date.now().toString(36),
      label: `Job ${(wfQueue['change_character']?.items?.length || 0) + 1}`,
      inputs: {
        ref_image: dsCharImagePath,
        ref_video: studioRefVideoPath,
        prompt: studioScenePrompt,
        aspect_ratio: studioFluxAspectRatio,
        bg_image: studioSelectedStage?.path || '',
        bg_prompt: studioBgPrompt,
      },
      filePaths: {},
      previews: {
        ref_image: dsCharImagePreview,
        ref_video: studioRefVideoPreview,
        bg_image: studioSelectedStage?.url || null,
      },
      gallerySelected: {},
      ytTitle,
      ytDescription: ytDesc,
      ytHashtags: ytHash,
      status: 'pending',
      progress: 0,
      outputVideo: null,
      error: null,
    };
    setWfQueue(prev => {
      const updated = {
        ...prev,
        change_character: {
          ...prev.change_character,
          items: [...(prev.change_character?.items || []), item],
          isProcessing: prev.change_character?.isProcessing || false,
        },
      };
      // Auto-start queue if not already processing
      if (!updated.change_character.isProcessing) {
        setTimeout(() => handleWfQueueStart('change_character'), 300);
      }
      return updated;
    });
    setStudioStatus(`\u2705 Added & starting: ${item.label}`);
  };

  // Dance Shorts - Add to Queue
  const handleDsQueueAdd = () => {
    if (!dsCharImagePath) return alert(t('dsNeedImage'));
    if (!studioRefVideoPath) return alert(t('studioNeedRefVideo'));
    const defaults = getStudioYtDefaults();
    const ytTitle = studioYtTitle.trim() || defaults.title;
    const ytDesc = studioYtDescription.trim() || defaults.description;
    const ytHash = studioYtHashtags.trim() || defaults.hashtags;
    const item = {
      id: Math.random().toString(36).slice(2) + Date.now().toString(36),
      label: `Job ${(wfQueue['change_character']?.items?.length || 0) + 1}`,
      inputs: {
        ref_image: dsCharImagePath,
        ref_video: studioRefVideoPath,
        prompt: studioScenePrompt,
        aspect_ratio: studioFluxAspectRatio,
        bg_image: studioSelectedStage?.path || '',
        bg_prompt: studioBgPrompt,
      },
      filePaths: {},
      previews: {
        ref_image: dsCharImagePreview,
        ref_video: studioRefVideoPreview,
        bg_image: studioSelectedStage?.url || null,
      },
      gallerySelected: {},
      ytTitle,
      ytDescription: ytDesc,
      ytHashtags: ytHash,
      status: 'pending',
      progress: 0,
      outputVideo: null,
      error: null,
    };
    setWfQueue(prev => ({
      ...prev,
      change_character: {
        ...prev.change_character,
        items: [...(prev.change_character?.items || []), item],
        isProcessing: prev.change_character?.isProcessing || false,
      },
    }));
    setStudioStatus(`\u2705 Added: ${item.label}`);
  };

  // Dance Shorts - Stage management
  const loadStudioStages = useCallback(async () => {
    try {
      const data = await listBackgrounds();
      setStudioStages(data.backgrounds || []);
    } catch (err) { console.error('Failed to load stages:', err); }
  }, []);

  const handleStudioStageDelete = async (filename) => {
    if (!window.confirm(t('studioStageDeleteConfirm'))) return;
    try {
      await deleteBackground(filename);
      setStudioStages(prev => prev.filter(s => s.filename !== filename));
      if (studioSelectedStage?.filename === filename) {
        setStudioSelectedStage(null);
      }
    } catch (err) { alert(`Delete failed: ${err.message}`); }
  };

  // Dance Shorts - YouTube defaults & upload
  const getStudioYtDefaults = useCallback(() => {
    const name = studioAvatarName || 'Avatar';
    const chName = studioYtChannelName || `Dancing ${name}`;
    const now = new Date();
    const dateStr = `${now.getFullYear()}:${String(now.getMonth() + 1).padStart(2, '0')}:${String(now.getDate()).padStart(2, '0')}:${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    return {
      title: `${name} #shorts - ${dateStr}`,
      description: `${chName} 많이 사랑해 주세요. 구독 좋아요 부탁드립니다.`,
      hashtags: `#${chName} #dancecover`,
    };
  }, [studioAvatarName, studioYtChannelName]);

  // Auto-fill YouTube metadata fields when avatar name or channel name changes
  useEffect(() => {
    const defaults = getStudioYtDefaults();
    setStudioYtTitle(defaults.title);
    setStudioYtDescription(defaults.description);
    setStudioYtHashtags(defaults.hashtags);
  }, [getStudioYtDefaults]);

  const handleStudioYtUpload = async () => {
    if (!studioOutputVideo) return;
    setStudioYtUploadStatus('uploading');
    const defaults = getStudioYtDefaults();
    const title = studioYtTitle.trim() || defaults.title;
    const description = studioYtDescription.trim() || defaults.description;
    const hashtags = studioYtHashtags.trim() || defaults.hashtags;
    try {
      const filename = studioOutputVideo.split('/').pop();
      const res = await uploadToYouTube(filename, title, description, hashtags);
      setStudioYtUploadStatus('success');
      setStudioYtUploadResult(res.youtube_url);
    } catch (err) {
      setStudioYtUploadStatus('error');
      setStudioYtUploadResult(err.response?.data?.detail || err.message || 'Upload failed');
    }
  };

  // Dance Shorts - Character image handlers
  const loadDsAvatarGroups = useCallback(async () => {
    try {
      const data = await listAvatarGroups();
      const groups = (data.groups || []).sort((a, b) => {
        if (a.toLowerCase() === 'yuna') return -1;
        if (b.toLowerCase() === 'yuna') return 1;
        return a.localeCompare(b);
      });
      setDsAvatarGroups(groups);
      if (groups.length > 0 && !dsAvatarSelectedGroup) {
        setDsAvatarSelectedGroup(groups[0]);
        setStudioAvatarName(groups[0]);
        setStudioYtChannelName(`Dancing ${groups[0]}`);
      }
    } catch (err) { console.error('Failed to load avatar groups:', err); }
  }, [dsAvatarSelectedGroup]);

  const loadDsAvatarImages = useCallback(async (group) => {
    if (!group) return;
    try {
      const data = await listAvatarImages(group);
      setDsAvatarImages(data.images || []);
    } catch (err) { console.error('Failed to load avatar images:', err); }
  }, []);


  const handleDsAvatarSelect = (img) => {
    setDsCharImagePath(img.path);
    setDsCharImagePreview(img.url);
    setStudioAvatarName(dsAvatarSelectedGroup);
    setStudioYtChannelName(`Dancing ${dsAvatarSelectedGroup}`);
  };

  const handleDsAvatarThumbDelete = async (img) => {
    if (!confirm(`Delete ${img.filename}?`)) return;
    try {
      await deleteAvatarImage(dsAvatarSelectedGroup, img.filename);
      await loadDsAvatarImages(dsAvatarSelectedGroup);
      if (dsCharImagePath === img.path) {
        setDsCharImagePath('');
        setDsCharImagePreview(null);
      }
    } catch (err) { alert(`Delete failed: ${err.message}`); }
  };

  // Keyboard arrow navigation: Left/Right for character images, Up/Down for stages
  useEffect(() => {
    if (activeMenu !== 'danceshorts') return;
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) return;
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        if (dsAvatarImages.length === 0) return;
        e.preventDefault();
        const currentIndex = dsAvatarImages.findIndex(img => img.path === dsCharImagePath);
        let newIndex;
        if (e.key === 'ArrowLeft') {
          newIndex = currentIndex <= 0 ? dsAvatarImages.length - 1 : currentIndex - 1;
        } else {
          newIndex = currentIndex >= dsAvatarImages.length - 1 ? 0 : currentIndex + 1;
        }
        handleDsAvatarSelect(dsAvatarImages[newIndex]);
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        if (studioStages.length === 0) return;
        e.preventDefault();
        const currentIndex = studioStages.findIndex(s => s.filename === studioSelectedStage?.filename);
        let newIndex;
        if (e.key === 'ArrowUp') {
          newIndex = currentIndex <= 0 ? studioStages.length - 1 : currentIndex - 1;
        } else {
          newIndex = currentIndex >= studioStages.length - 1 ? 0 : currentIndex + 1;
        }
        setStudioSelectedStage(studioStages[newIndex]);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [activeMenu, dsAvatarImages, dsCharImagePath, studioStages, studioSelectedStage]);

  // Avatar popup: navigate to prev/next image
  const avatarPopupNavigate = useCallback((direction) => {
    if (!avatarPopup) return;
    const images = avatarPopup.source === 'danceshorts' ? dsAvatarImages
      : (workflowStates[avatarPopup.wfId]?.avatarImages?.[avatarPopup.group] || []);
    if (images.length <= 1) return;
    const idx = images.findIndex(i => i.filename === avatarPopup.filename);
    if (idx === -1) return;
    const next = direction === 'right'
      ? images[(idx + 1) % images.length]
      : images[(idx - 1 + images.length) % images.length];
    setAvatarPopup(prev => ({ ...prev, url: next.url, filename: next.filename, img: next }));
    if (avatarPopup.source === 'danceshorts') {
      setDsCharImagePath(next.path);
      setDsCharImagePreview(next.url);
    }
  }, [avatarPopup, dsAvatarImages, workflowStates]);

  // Avatar popup: Left/Right arrow key browsing
  useEffect(() => {
    if (!avatarPopup) return;
    const handler = (e) => {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      e.preventDefault();
      avatarPopupNavigate(e.key === 'ArrowRight' ? 'right' : 'left');
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [avatarPopup, avatarPopupNavigate]);

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

  // Dance Shorts - Video Timeline handlers
  const handleDsVideoMeta = (e) => {
    const dur = e.target.duration;
    if (dur && isFinite(dur)) {
      const rounded = Math.round(dur * 100) / 100;
      setDsVideoDuration(rounded);
      setDsTrimStart(0);
      setDsTrimEnd(rounded);
      setDsPlayheadPosition(0);
    }
  };

  const handleDsVideoTimeUpdate = useCallback((e) => {
    setDsPlayheadPosition(Math.round(e.target.currentTime * 100) / 100);
  }, []);

  const dsStateRef = useRef({ duration: 0, trimStart: 0, trimEnd: 0 });
  useEffect(() => {
    dsStateRef.current = { duration: dsVideoDuration, trimStart: dsTrimStart, trimEnd: dsTrimEnd };
  }, [dsVideoDuration, dsTrimStart, dsTrimEnd]);

  const handleDsTimelineDragStart = useCallback((handleType, e) => {
    e.preventDefault();
    e.stopPropagation();
    const timelineEl = dsTimelineRef.current;
    if (!timelineEl) return;

    const onMouseMove = (moveEvent) => {
      const rect = timelineEl.getBoundingClientRect();
      const x = Math.max(0, Math.min(moveEvent.clientX - rect.left, rect.width));
      const { duration, trimStart, trimEnd } = dsStateRef.current;
      if (!duration) return;
      const time = Math.round(((x / rect.width) * duration) * 100) / 100;
      if (handleType === 'start') {
        const newStart = Math.max(0, Math.min(time, trimEnd - 0.1));
        setDsTrimStart(newStart);
        setDsPlayheadPosition(newStart);
        if (dsVideoRef.current) dsVideoRef.current.currentTime = newStart;
      } else {
        const newEnd = Math.min(duration, Math.max(time, trimStart + 0.1));
        setDsTrimEnd(newEnd);
        setDsPlayheadPosition(newEnd);
        if (dsVideoRef.current) dsVideoRef.current.currentTime = newEnd;
      }
    };
    const onMouseUp = () => {
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, []);

  const handleDsRulerClick = useCallback((e) => {
    const timelineEl = dsTimelineRef.current;
    if (!timelineEl) return;
    const rect = timelineEl.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    const { duration } = dsStateRef.current;
    if (!duration) return;
    const time = Math.round(((x / rect.width) * duration) * 100) / 100;
    if (dsVideoRef.current) dsVideoRef.current.currentTime = time;
    setDsPlayheadPosition(time);
  }, []);

  const handleDsTrim = async () => {
    if (!studioRefVideoPath || !dsVideoDuration) return;
    if (dsTrimEnd <= dsTrimStart) return;
    setDsTrimming(true);
    try {
      const result = await trimVideo(studioRefVideoPath, dsTrimStart, dsTrimEnd);
      setStudioRefVideoPath(result.path);
      setStudioRefVideoPreview(result.url);
      setDsVideoDuration(result.duration);
      setDsTrimStart(0);
      setDsTrimEnd(result.duration);
      setDsPlayheadPosition(0);
    } catch (err) {
      alert(`Trim error: ${err.message}`);
    } finally {
      setDsTrimming(false);
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

  const handleGalleryUpload = async (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    setGalleryUploading(true);
    setGalleryUploadProgress(0);
    for (let i = 0; i < files.length; i++) {
      try {
        const item = await uploadToGallery(files[i], (pct) => {
          setGalleryUploadProgress(Math.round(((i * 100) + pct) / files.length));
        });
        if (item.type === 'image') {
          setGalleryImages(prev => [item, ...prev]);
        } else {
          setVideos(prev => [item, ...prev]);
        }
      } catch (err) {
        console.error('Gallery upload failed:', err);
      }
    }
    setGalleryUploading(false);
    setGalleryUploadProgress(0);
    if (galleryUploadRef.current) galleryUploadRef.current.value = '';
  };

  const handleRegisterAsAvatar = async (img) => {
    try {
      const groupsData = await listAvatarGroups();
      const existing = groupsData.groups || [];
      const hint = existing.length > 0 ? `\n(${existing.join(', ')})` : '';
      const defaultGroup = existing.includes('yuna') ? 'yuna' : (existing[0] || 'yuna');
      const group = window.prompt(`Avatar group name:${hint}`, defaultGroup);
      if (!group) return;

      // Direct registration (copy image to avatar directory)
      await registerAvatar(img.path, group.trim());

      // Refresh avatar groups in all workflows + Dance Shorts
      const refreshed = await listAvatarGroups();
      const newGroups = refreshed.groups || [];
      setWorkflowStates(prev => {
        const next = { ...prev };
        for (const wfId of Object.keys(next)) {
          next[wfId] = { ...next[wfId], avatarGroups: newGroups, avatarImages: {} };
        }
        return next;
      });
      // Refresh Dance Shorts avatar list
      await loadDsAvatarGroups();
      if (dsAvatarSelectedGroup) await loadDsAvatarImages(dsAvatarSelectedGroup);
      alert(`Avatar registered in "${group.trim()}"`);
    } catch (err) { console.error('Failed to register avatar:', err); alert('Failed: ' + (err.response?.data?.detail || err.message)); }
  };

  const handleRegisterAsStage = async (img) => {
    try {
      const resp = await fetch(img.url);
      const blob = await resp.blob();
      const file = new File([blob], img.filename, { type: blob.type });
      await uploadBackground(file);
      await loadStudioStages();
      alert(`Stage registered: ${img.filename}`);
    } catch (err) { alert(`Failed: ${err.message}`); }
  };

  const handleGalleryStageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      await uploadBackground(file);
      await loadStudioStages();
    } catch (err) { alert(`Upload failed: ${err.message}`); }
    e.target.value = '';
  };

  const handleYtUploadOpen = (filename) => {
    setYtUploadTarget(filename);
    setYtUploadForm({ title: '', description: '', hashtags: '' });
    setYtUploadStatus(null);
    setYtUploadResult(null);
  };

  const handleYtUploadClose = () => {
    setYtUploadTarget(null);
    setYtUploadStatus(null);
    setYtUploadResult(null);
  };

  const handleYtUploadSubmit = async () => {
    if (!ytUploadTarget) return;
    setYtUploadStatus('uploading');
    try {
      const res = await uploadToYouTube(ytUploadTarget, ytUploadForm.title, ytUploadForm.description, ytUploadForm.hashtags);
      setYtUploadStatus('success');
      setYtUploadResult(res.youtube_url);
    } catch (err) {
      setYtUploadStatus('error');
      setYtUploadResult(err.response?.data?.detail || err.message || 'Upload failed');
    }
  };

  useEffect(() => { if (activeMenu === 'gallery') fetchGallery(); }, [activeMenu, fetchGallery]);

  // Load stages and avatars when Dance Shorts menu is active
  useEffect(() => {
    if (activeMenu === 'danceshorts') {
      loadStudioStages();
      loadDsAvatarGroups();
    }
  }, [activeMenu, loadStudioStages, loadDsAvatarGroups]);

  // Load avatar images when group changes
  useEffect(() => {
    if (activeMenu === 'danceshorts' && dsAvatarSelectedGroup) {
      loadDsAvatarImages(dsAvatarSelectedGroup);
    }
  }, [activeMenu, dsAvatarSelectedGroup, loadDsAvatarImages]);

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
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    if (files.length === 1) {
      // Single file: existing behavior
      updateWfPreview(wfId, key, URL.createObjectURL(files[0]));
      try {
        const data = await uploadImage(files[0]);
        updateWfFilePath(wfId, key, data.path);
      } catch (err) {
        updateWfState(wfId, { status: `Upload error: ${err.message}` });
      }
    } else {
      // Multi-file: upload all, store paths in multiUploads
      updateWfState(wfId, { status: `Uploading ${files.length} images...` });
      try {
        const uploaded = [];
        for (const file of files) {
          const data = await uploadImage(file);
          uploaded.push({ path: data.path, url: data.url, name: file.name });
        }
        // Set first as current, store all in multiUploads
        updateWfFilePath(wfId, key, uploaded[0].path);
        updateWfPreview(wfId, key, uploaded[0].url);
        setWorkflowStates(prev => ({
          ...prev, [wfId]: {
            ...prev[wfId],
            multiUploads: { ...prev[wfId].multiUploads, [key]: uploaded },
          },
        }));
        updateWfState(wfId, { status: `✅ ${uploaded.length} images uploaded` });
      } catch (err) {
        updateWfState(wfId, { status: `Upload error: ${err.message}` });
      }
    }
    // Reset input so same files can be re-selected
    e.target.value = '';
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
        const defaultGroup = groups.includes('yuna') ? 'yuna' : groups[0];
        handleWfAvatarSelect(wfId, defaultGroup);
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

  const handleWfAvatarDelete = async (wfId, group, img) => {
    if (!window.confirm(`Delete "${img.filename}" from ${group}?`)) return;
    try {
      await deleteAvatarImage(group, img.filename);
      // Refresh avatar images for ALL workflows that share this group
      const data = await listAvatarImages(group);
      const updatedImages = data.images || [];
      setWorkflowStates(prev => {
        const next = { ...prev };
        for (const wId of Object.keys(next)) {
          if (next[wId]?.avatarImages?.[group]) {
            next[wId] = {
              ...next[wId],
              avatarImages: { ...next[wId].avatarImages, [group]: updatedImages },
            };
            // Clear selection if the deleted image was selected
            if (next[wId].filePaths) {
              for (const key of Object.keys(next[wId].filePaths)) {
                if (next[wId].filePaths[key] === img.path) {
                  next[wId] = {
                    ...next[wId],
                    filePaths: { ...next[wId].filePaths, [key]: '' },
                    previews: { ...next[wId].previews, [key]: '' },
                  };
                }
              }
            }
          }
        }
        return next;
      });
      // If the group is now empty, refresh groups for all workflows
      if (updatedImages.length === 0) {
        const groupData = await listAvatarGroups();
        const groups = groupData.groups || [];
        setWorkflowStates(prev => {
          const next = { ...prev };
          for (const wId of Object.keys(next)) {
            if (next[wId]?.avatarGroups) {
              next[wId] = { ...next[wId], avatarGroups: groups };
            }
          }
          return next;
        });
      }
    } catch (err) {
      console.error('Failed to delete avatar:', err);
    }
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

  const handleWfFashionMultiToggle = (wfId, style) => {
    setWorkflowStates(prev => {
      const wf = prev[wfId];
      const current = wf.fashionMultiSelected || [];
      const exists = current.some(s => s.id === style.id);
      const updated = exists ? current.filter(s => s.id !== style.id) : [...current, style];
      return {
        ...prev, [wfId]: {
          ...wf,
          fashionMultiSelected: updated,
          // Also apply last selected to prompt for preview
          inputs: { ...wf.inputs, fashion_prompt: updated.length > 0 ? updated[updated.length - 1].prompt : wf.inputs.fashion_prompt },
        },
      };
    });
  };

  const handleWfBatchQueue = (wfId) => {
    const wfState = workflowStates[wfId];
    const wfDef = workflows.find(w => w.id === wfId);
    if (!wfDef || !wfState) return;
    const selectedStyles = wfState.fashionMultiSelected || [];

    // Check for multi-uploaded images
    const imageInputKeys = wfDef.inputs.filter(i => i.type === 'image' && !i.avatar_gallery).map(i => i.key);
    let multiImages = [];
    for (const key of imageInputKeys) {
      const uploads = wfState.multiUploads?.[key];
      if (uploads && uploads.length > 1) {
        multiImages = uploads.map(u => ({ key, path: u.path, url: u.url, name: u.name }));
        break;
      }
    }

    // Need at least one dimension to batch
    if (selectedStyles.length === 0 && multiImages.length === 0) return;

    // Validate required inputs (skip keys that will be overridden by multiUploads)
    const multiUploadKeys = new Set(multiImages.map(m => m.key));
    for (const inp of wfDef.inputs) {
      if (inp.required && inp.key !== 'fashion_prompt' && !multiUploadKeys.has(inp.key)) {
        const val = (inp.type === 'image' || inp.type === 'video')
          ? wfState.filePaths[inp.key]
          : wfState.inputs[inp.key];
        if (!val) {
          const msg = `Required: ${inp.label[lang] || inp.key}`;
          updateWfState(wfId, { status: msg });
          window.alert(msg);
          return;
        }
      }
    }

    // Build combinations: images × styles
    const imageSets = multiImages.length > 0 ? multiImages : [null];
    const styleSets = selectedStyles.length > 0
      ? selectedStyles
      : [null]; // null = use current inputs as-is (no style override)

    const newItems = [];
    for (const img of imageSets) {
      for (const style of styleSets) {
        const imgLabel = img ? `[${img.name.replace(/\.[^.]+$/, '')}]` : '';
        const styleLabel = style ? (style.category ? `${style.category} - ` : '') + (style.prompt.length > 40 ? style.prompt.slice(0, 40) + '...' : style.prompt) : '';
        const label = [imgLabel, styleLabel].filter(Boolean).join(' ');

        const itemInputs = { ...wfState.inputs };
        if (style) itemInputs.fashion_prompt = style.prompt;

        const item = {
          id: Math.random().toString(36).slice(2) + Date.now().toString(36) + newItems.length,
          label: label || `Job ${newItems.length + 1}`,
          inputs: itemInputs,
          filePaths: { ...wfState.filePaths },
          previews: { ...wfState.previews },
          gallerySelected: { ...wfState.gallerySelected },
          ytTitle: '', ytDescription: '', ytHashtags: '',
          status: 'pending', progress: 0, outputVideo: null, error: null,
        };
        if (img) {
          item.filePaths = { ...item.filePaths, [img.key]: img.path };
          item.previews = { ...item.previews, [img.key]: img.url };
        }
        newItems.push(item);
      }
    }

    // Add all to queue
    setWfQueue(prev => ({
      ...prev,
      [wfId]: {
        ...prev[wfId],
        items: [...(prev[wfId]?.items || []), ...newItems],
        isProcessing: prev[wfId]?.isProcessing || false,
      },
    }));

    const parts = [];
    if (multiImages.length > 0) parts.push(`${multiImages.length} images`);
    if (selectedStyles.length > 0) parts.push(`${selectedStyles.length} styles`);
    const totalLabel = `✅ ${newItems.length} items (${parts.join(' × ')}) added`;

    // Clear multi-selection
    updateWfState(wfId, {
      fashionMultiSelected: [],
      status: totalLabel,
    });

    // Auto-start queue
    setTimeout(() => handleWfQueueStart(wfId), 300);
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

  const handleWfVideoClear = useCallback((wfId, key) => {
    setWorkflowStates(prev => {
      const wf = prev[wfId];
      if (!wf) return prev;
      const newFilePaths = { ...wf.filePaths };
      const newPreviews = { ...wf.previews };
      const newDuration = { ...wf.videoDuration };
      const newTrimStart = { ...wf.trimStart };
      const newTrimEnd = { ...wf.trimEnd };
      const newPlayhead = { ...wf.playheadPosition };
      delete newFilePaths[key];
      delete newPreviews[key];
      delete newDuration[key];
      delete newTrimStart[key];
      delete newTrimEnd[key];
      delete newPlayhead[key];
      return { ...prev, [wfId]: { ...wf,
        filePaths: newFilePaths, previews: newPreviews, videoDuration: newDuration,
        trimStart: newTrimStart, trimEnd: newTrimEnd, playheadPosition: newPlayhead,
      }};
    });
  }, []);

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

  // Generate via queue: add item + auto-start (for change_character)
  const handleWfGenerateViaQueue = (wfId) => {
    handleWfQueueAdd(wfId);
    setTimeout(() => {
      const q = wfQueueRef.current[wfId];
      if (q && !q.isProcessing) handleWfQueueStart(wfId);
    }, 300);
  };

  const handleWfGenerate = async (wfId) => {
    // Route workflows through queue for persistence and sequential execution
    if (wfId === 'change_character' || wfId === 'face_swap') return handleWfGenerateViaQueue(wfId);

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
      saveActiveTask(wfId, taskId);
      const poll = setInterval(async () => {
        try {
          const s = await getTaskStatus(taskId);
          updateWfState(wfId, {
            progress: Math.round((s.progress || 0) * 100),
            status: s.status_message || s.status,
          });
          if (s.status === 'completed') {
            clearInterval(poll);
            removeActiveTask(wfId);
            updateWfState(wfId, { isGenerating: false, progress: 100, outputVideo: s.output_url, currentTaskId: null });
            // Refresh avatar gallery for all workflows after fashion_change saves output to avatars
            if (wfId === 'fashion_change') {
              Object.keys(workflowStates).forEach(id => {
                setWorkflowStates(prev => ({ ...prev, [id]: { ...prev[id], avatarGroups: [], avatarImages: {} } }));
              });
            }
          } else if (s.status === 'failed') {
            clearInterval(poll);
            removeActiveTask(wfId);
            updateWfState(wfId, { isGenerating: false, status: `Error: ${s.error || 'Failed'}`, currentTaskId: null });
          } else if (s.status === 'cancelled') {
            clearInterval(poll);
            removeActiveTask(wfId);
            updateWfState(wfId, { isGenerating: false, status: t('wfCancelled'), currentTaskId: null });
          }
        } catch (err) {
          clearInterval(poll);
          removeActiveTask(wfId);
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
    removeActiveTask(wfId);
    updateWfState(wfId, { isGenerating: false, status: t('wfCancelled'), currentTaskId: null });
  };

  // ─── Queue handlers ───
  // Keep ref in sync for use inside async loops
  useEffect(() => {
    wfQueueRef.current = wfQueue;
    saveQueueToStorage(wfQueue);
  }, [wfQueue]);

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
          const msg = `Required: ${inp.label[lang] || inp.key}`;
          updateWfState(wfId, { status: msg });
          window.alert(msg);
          return;
        }
      }
    }

    // Build a descriptive label
    let queueLabel = `Job ${(wfQueue[wfId]?.items?.length || 0) + 1}`;
    if (wfId === 'fashion_change' && wfState.inputs?.fashion_prompt) {
      const kw = wfState.inputs.fashion_prompt.length > 40
        ? wfState.inputs.fashion_prompt.slice(0, 40) + '...'
        : wfState.inputs.fashion_prompt;
      queueLabel += ` - ${kw}`;
    }

    const item = {
      id: Math.random().toString(36).slice(2) + Date.now().toString(36),
      label: queueLabel,
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
    updateWfState(wfId, { status: `\u2705 Added: ${item.label}` });
  };

  const handleWfQueueStart = async (wfId) => {
    const queue = wfQueueRef.current[wfId];
    if (!queue?.items?.length || queue.isProcessing) return;

    setWfQueue(prev => ({ ...prev, [wfId]: { ...prev[wfId], isProcessing: true } }));

    const wfDef = workflows.find(w => w.id === wfId);

    // Process pending items, including any newly added during execution
    const processNextPending = async () => {
      const currentQueue = wfQueueRef.current[wfId];
      if (!currentQueue) return null;
      return currentQueue.items.find(i => i.status === 'pending');
    };

    let nextItem = await processNextPending();
    while (nextItem) {
      const itemId = nextItem.id;
      updateQueueItem(wfId, itemId, { status: 'running', progress: 0 });

      // Re-read from ref to get latest data
      const currentItem = wfQueueRef.current[wfId]?.items?.find(i => i.id === itemId);
      if (!currentItem) { nextItem = await processNextPending(); continue; }

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
          yt_upload: !!(currentItem.ytTitle || currentItem.ytDescription || currentItem.ytHashtags),
        });
        const taskId = data.task_id;
        updateQueueItem(wfId, itemId, { taskId });

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
                  outputVideo: s.output_url || s.output_path, taskId: null,
                });
                resolve();
              } else if (s.status === 'failed') {
                clearInterval(poll);
                updateQueueItem(wfId, itemId, {
                  status: 'failed', error: s.message || 'Failed', taskId: null,
                });
                resolve();
              } else if (s.status === 'cancelled') {
                clearInterval(poll);
                updateQueueItem(wfId, itemId, {
                  status: 'failed', error: 'Cancelled', taskId: null,
                });
                resolve();
              }
            } catch (err) {
              clearInterval(poll);
              updateQueueItem(wfId, itemId, { status: 'failed', error: err.message, taskId: null });
              resolve();
            }
          }, 3000);
        });
      } catch (err) {
        updateQueueItem(wfId, itemId, { status: 'failed', error: err.message });
      }

      // Check for next pending item (including newly added ones)
      nextItem = await processNextPending();
    }

    // Queue done: remove completed/failed items from storage (keep in UI briefly)
    setWfQueue(prev => {
      const q = prev[wfId];
      if (!q) return { ...prev, [wfId]: { isProcessing: false, items: [] } };
      return { ...prev, [wfId]: { ...q, isProcessing: false } };
    });
    // Auto-clear completed/failed after 10 seconds
    setTimeout(() => {
      setWfQueue(prev => {
        const q = prev[wfId];
        if (!q) return prev;
        const remaining = q.items.filter(i => i.status === 'pending' || i.status === 'running');
        if (remaining.length === q.items.length) return prev;
        return { ...prev, [wfId]: { ...q, items: remaining } };
      });
    }, 10000);
  };

  const handleWfQueueRemove = async (wfId, itemId) => {
    const q = wfQueueRef.current[wfId];
    const item = q?.items?.find(i => i.id === itemId);
    if (item?.status === 'running' && item?.taskId) {
      try { await cancelGeneration(item.taskId); } catch {}
    }
    setWfQueue(prev => {
      const q = prev[wfId];
      if (!q) return prev;
      return { ...prev, [wfId]: { ...q, items: q.items.filter(i => i.id !== itemId) } };
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
                <div className="avatar-viewer" style={{ cursor: 'pointer' }}
                  onClick={() => {
                    const selectedImg = avatarImgs.find(img => img.url === wfState.previews[inputDef.key]);
                    setAvatarPopup({
                      url: wfState.previews[inputDef.key],
                      filename: selectedImg?.filename || '',
                      group: currentGroup,
                      wfId,
                      img: selectedImg,
                    });
                  }}>
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
                      <button className="avatar-thumb-delete" title="Delete"
                        onClick={e => { e.stopPropagation(); handleWfAvatarDelete(wfId, currentGroup, img); }}>
                        &times;
                      </button>
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
        {
          const multiUploads = wfState.multiUploads?.[inputDef.key] || [];
          return (
            <div className="card" key={inputDef.key}>
              <h3>{label}{multiUploads.length > 1 ? ` (${multiUploads.length} images)` : ''}</h3>
              <div className="drop-zone" onClick={() => document.getElementById(inputIdBase)?.click()}
                onDragOver={e => e.preventDefault()}
                onDrop={e => {
                  e.preventDefault();
                  const files = e.dataTransfer.files;
                  if (files.length > 0) {
                    const inp = document.getElementById(inputIdBase);
                    inp.files = files;
                    inp.dispatchEvent(new Event('change', { bubbles: true }));
                  }
                }}>
                {wfState.previews[inputDef.key]
                  ? <img src={wfState.previews[inputDef.key]} alt="" style={{ maxHeight: inputDef.large_viewer ? 400 : 200, objectFit: 'contain' }} />
                  : <p>{t('dropImageHere')}</p>}
              </div>
              <input id={inputIdBase} type="file" accept="image/*" multiple style={{ display: 'none' }} onChange={e => handleWfImageUpload(wfId, inputDef.key, e)} />
              {multiUploads.length > 1 && (
                <>
                  <div className="multi-image-preview" style={{ display: 'flex', gap: 4, marginTop: 6, flexWrap: 'wrap' }}>
                    {multiUploads.map((img, i) => (
                      <img key={i} src={img.url} alt={img.name} style={{ height: 50, objectFit: 'contain', borderRadius: 4, border: '1px solid var(--border)' }} />
                    ))}
                  </div>
                  <button className="btn primary" style={{ width: '100%', marginTop: 8 }}
                    onClick={() => handleWfBatchQueue(wfId)}>
                    Add {multiUploads.length} to Queue & Start
                  </button>
                </>
              )}
            </div>
          );
        }
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
                    <button className="btn btn-youtube-dl" onClick={() => handleWfYoutubeDownload(wfId, inputDef.key)}
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
                    <button className="btn secondary trim-btn" style={{ width: '100%', marginTop: 6 }}
                      onClick={() => handleWfVideoClear(wfId, inputDef.key)}>
                      {t('wfClearVideo')}
                    </button>
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

      case 'select': {
        const currentVal = wfState.inputs[inputDef.key] ?? inputDef.default ?? '';
        const selectedOpt = (inputDef.options || []).find(o => o.value === currentVal);
        const selectedLabel = selectedOpt ? (selectedOpt.label[lang] || selectedOpt.label.en || selectedOpt.value) : currentVal;
        return (
          <div className="card" key={inputDef.key}>
            <h3>{label}: {selectedLabel}</h3>
            <div className="form-group">
              <select value={currentVal} onChange={e => updateWfInput(wfId, inputDef.key, e.target.value)}
                style={{ width: '100%', padding: '8px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-secondary)', color: 'var(--text-primary)', fontSize: 14 }}>
                {(inputDef.options || []).map(opt => (
                  <option key={opt.value} value={opt.value}>{opt.label[lang] || opt.label.en || opt.value}</option>
                ))}
              </select>
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
        const multiSelected = wfState.fashionMultiSelected || [];
        const multiCount = multiSelected.length;
        return (
          <div className="card" key={inputDef.key}>
            <h3>{t('fashionStyle')}{multiCount > 0 ? ` (${multiCount} selected)` : ''}</h3>
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
              {multiCount > 0 && (
                <button className="btn secondary fashion-random-btn" onClick={() => updateWfState(wfId, { fashionMultiSelected: [] })}>
                  Clear ({multiCount})
                </button>
              )}
            </div>
            {multiCount > 0 && (
              <button className="btn primary" style={{ width: '100%', marginBottom: 8 }}
                onClick={() => handleWfBatchQueue(wfId)}>
                🚀 Add {multiCount} to Queue & Start
              </button>
            )}
            <div className="fashion-grid">
              {filtered.map(style => {
                const isMultiSelected = multiSelected.some(s => s.id === style.id);
                const isSingleSelected = wfState.inputs?.fashion_prompt === style.prompt && !isMultiSelected;
                return (
                  <div key={style.id} className={`fashion-item${isMultiSelected ? ' selected' : isSingleSelected ? ' selected' : ''}`}
                    onClick={() => handleWfFashionMultiToggle(wfId, style)}>
                    {isMultiSelected && <span style={{ position: 'absolute', top: 2, right: 6, fontSize: 14 }}>✓</span>}
                    <span className="fashion-item-cat">{style.category}</span>
                    <span className="fashion-item-text">{style.prompt}</span>
                  </div>
                );
              })}
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
            className={`sidebar-item${activeMenu === 'danceshorts' ? ' active' : ''}`}
            onClick={() => setActiveMenu('danceshorts')}
          >
            <span className="sidebar-icon">&#128131;</span>
            {t('menuDanceShorts')}
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
            <span className="sidebar-icon">&#127916;</span>
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

                        {/* YouTube Shorts metadata — only for video-output workflows (not change_character, handled in Dance Shorts) */}
                        {wf.output_type !== 'image' && wf.id !== 'change_character' && <div className="card yt-meta-card">
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
                        </div>}

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
                              className={`btn ${(wf.id === 'change_character' || wf.id === 'fashion_change' || wf.id === 'face_swap') ? 'generate-btn-green' : 'generate-btn'}`}
                              onClick={() => handleWfGenerate(wf.id)}
                              style={{ flex: 1 }}
                            >
                              {t('wfGenerateBtn')}
                            </button>
                          )}
                          <button
                            className="btn secondary"
                            onClick={() => handleWfQueueAdd(wf.id)}
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
                                  <div key={item.id} className={`queue-item queue-item--${item.status}${(item.previews?.ref_image || item.previews?.ref_video || item.previews?.bg_image) ? ' queue-item--rich' : ''}`}>
                                    <div className="queue-item-top">
                                      {/* Thumbnails */}
                                      {(item.previews?.ref_image || item.previews?.ref_video || item.previews?.bg_image) && (
                                        <div className="queue-item-thumbs">
                                          {item.previews?.ref_image && (
                                            <img src={item.previews.ref_image} alt="avatar" className="queue-thumb queue-thumb--avatar" />
                                          )}
                                          {item.previews?.ref_video && (
                                            <video src={item.previews.ref_video} className="queue-thumb queue-thumb--video" muted preload="metadata" />
                                          )}
                                          {item.previews?.bg_image && (
                                            <img src={item.previews.bg_image} alt="bg" className="queue-thumb queue-thumb--bg" />
                                          )}
                                        </div>
                                      )}
                                      <div className="queue-item-info">
                                        <div className="queue-item-title-row">
                                          <span className="queue-item-status">
                                            {item.status === 'completed' ? '\u2705' : item.status === 'running' ? '\u23f3' : item.status === 'failed' ? '\u274c' : '\u23f8'}
                                          </span>
                                          <span className="queue-item-label" title={item.ytTitle || item.inputs?.prompt || item.inputs?.fashion_prompt || ''}>
                                            {item.ytTitle || item.label}
                                          </span>
                                          <button className="queue-item-remove" onClick={() => handleWfQueueRemove(wf.id, item.id)}>&times;</button>
                                        </div>
                                        {item.ytTitle && item.label && (
                                          <div className="queue-item-sublabel">{item.label}</div>
                                        )}
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
                                      </div>
                                    </div>
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

          {/* ============ DANCE SHORTS ============ */}
          {activeMenu === 'danceshorts' && (
            <div className="page-content">
              <div className="two-column">
                {/* Left Column - Settings */}
                <div className="column">
                  <div className="card">
                    <h3>{t('menuDanceShorts')}</h3>

                    {/* Character Image - Avatar Gallery */}
                    <div className="form-group">
                      <label>{t('dsCharImage')}</label>
                      {dsAvatarGroups.length > 0 && (
                        <div className="sub-tabs" style={{ marginBottom: 8 }}>
                          {dsAvatarGroups.map(g => (
                            <button key={g} className={dsAvatarSelectedGroup === g ? 'active' : ''}
                              onClick={() => { setDsAvatarSelectedGroup(g); setStudioAvatarName(g); setStudioYtChannelName(`Dancing ${g}`); }}>{g}</button>
                          ))}
                        </div>
                      )}
                      <div className="stage-gallery avatar-thumb-gallery">
                        {dsAvatarImages.map(img => (
                          <div key={img.filename}
                            className={`stage-item avatar-thumb-item${dsCharImagePath === img.path ? ' selected' : ''}`}
                            onClick={() => handleDsAvatarSelect(img)}>
                            <img src={img.url} alt={img.filename} />
                            <button className="stage-delete-btn"
                              onClick={e => { e.stopPropagation(); handleDsAvatarThumbDelete(img); }}
                              title={t('galleryDelete')}>×</button>
                          </div>
                        ))}
                      </div>
                      {dsCharImagePreview && (
                        <div className="ds-char-viewer" onDoubleClick={() => {
                          const img = dsAvatarImages.find(i => i.path === dsCharImagePath);
                          if (img) setAvatarPopup({ url: img.url, filename: img.filename, group: dsAvatarSelectedGroup, img, source: 'danceshorts' });
                        }} title="Double-click to enlarge">
                          <img src={dsCharImagePreview} alt="Character" />
                        </div>
                      )}
                    </div>

                    {/* Avatar Name */}
                    <div className="form-group">
                      <label>{t('studioAvatarName')}</label>
                      <input type="text" value={studioAvatarName}
                        onChange={e => setStudioAvatarName(e.target.value)}
                        placeholder="e.g. Lina, Yuna..." />
                    </div>

                    {/* Stage Selection */}
                    <div className="form-group">
                      <label>{t('studioStageSelect')}</label>
                      <div className="stage-gallery">
                        {studioStages.map(stage => (
                          <div key={stage.filename}
                            className={`stage-item${studioSelectedStage?.filename === stage.filename ? ' selected' : ''}`}
                            onClick={() => setStudioSelectedStage(stage)}>
                            <img src={stage.url} alt={stage.filename} />
                            <button className="stage-delete-btn"
                              onClick={e => { e.stopPropagation(); handleStudioStageDelete(stage.filename); }}
                              title={t('galleryDelete')}>×</button>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Background Effect */}
                    <div className="form-group">
                      <label>{t('studioBgEffect')}</label>
                      <textarea value={studioBgPrompt} onChange={e => setStudioBgPrompt(e.target.value)} rows={2}
                        placeholder="e.g. neon lights, foggy atmosphere, colorful stage lighting..." />
                    </div>

                    {/* Scene Description */}
                    <div className="form-group">
                      <label>{t('studioSceneDesc')}</label>
                      <textarea value={studioScenePrompt} onChange={e => setStudioScenePrompt(e.target.value)} rows={2} />
                    </div>

                    {/* Ref Video Source */}
                    <div className="sub-tabs" style={{ marginBottom: 12 }}>
                      <button className={studioRefVideoMode === 'youtube' ? 'active' : ''}
                        onClick={() => setStudioRefVideoMode('youtube')}>{t('wfVideoYoutube')}</button>
                      <button className={studioRefVideoMode === 'upload' ? 'active' : ''}
                        onClick={() => setStudioRefVideoMode('upload')}>{t('wfVideoUpload')}</button>
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
                        <button className="btn btn-youtube-dl" disabled={studioYoutubeDownloading}
                          onClick={studioHandleYoutubeDownload}>
                          {studioYoutubeDownloading ? t('wfDownloading') : t('wfDownloadBtn')}
                        </button>
                      </div>
                    )}

                    {studioRefVideoPreview && (
                      <div className="video-editor-card" style={{ marginTop: 8 }}>
                        <video ref={dsVideoRef} controls src={studioRefVideoPreview}
                          style={{ width: '100%', borderRadius: 8 }}
                          onLoadedMetadata={handleDsVideoMeta}
                          onTimeUpdate={handleDsVideoTimeUpdate} />
                        {dsVideoDuration > 0 && (
                          <>
                            <div className="video-timeline-container" style={{ marginTop: 8 }}>
                              <div className="video-timeline-ruler" onClick={handleDsRulerClick}>
                                {generateRulerTicks(dsVideoDuration).map(tick => (
                                  <div key={tick} className="ruler-tick"
                                    style={{ left: `${(tick / dsVideoDuration) * 100}%` }}>
                                    <span>{formatTime(tick)}</span>
                                  </div>
                                ))}
                                <div className="timeline-playhead"
                                  style={{ left: `${(dsPlayheadPosition / dsVideoDuration) * 100}%` }} />
                              </div>
                              <div className="video-timeline-track" ref={dsTimelineRef}>
                                <div className="timeline-region-dimmed"
                                  style={{ left: 0, width: `${(dsTrimStart / dsVideoDuration) * 100}%` }} />
                                <div className="timeline-region-active"
                                  style={{ left: `${(dsTrimStart / dsVideoDuration) * 100}%`,
                                    width: `${((dsTrimEnd - dsTrimStart) / dsVideoDuration) * 100}%` }} />
                                <div className="timeline-region-dimmed"
                                  style={{ left: `${(dsTrimEnd / dsVideoDuration) * 100}%`,
                                    width: `${((dsVideoDuration - dsTrimEnd) / dsVideoDuration) * 100}%` }} />
                                <div className="timeline-handle timeline-handle-left"
                                  style={{ left: `${(dsTrimStart / dsVideoDuration) * 100}%` }}
                                  onMouseDown={(e) => handleDsTimelineDragStart('start', e)} />
                                <div className="timeline-handle timeline-handle-right"
                                  style={{ left: `${(dsTrimEnd / dsVideoDuration) * 100}%` }}
                                  onMouseDown={(e) => handleDsTimelineDragStart('end', e)} />
                              </div>
                            </div>
                            <div className="video-editor-controls" style={{ marginTop: 8 }}>
                              <div className="trim-controls-vertical">
                                <div className="trim-duration">{t('wfDuration')}: {formatTime(dsVideoDuration)}</div>
                                <label>Start:
                                  <input type="number" className="trim-input" min={0} max={dsVideoDuration} step={0.1}
                                    value={dsTrimStart} onChange={e => {
                                      const v = parseFloat(e.target.value) || 0;
                                      setDsTrimStart(Math.max(0, Math.min(v, dsTrimEnd - 0.1)));
                                    }} />
                                  <span className="trim-unit">s</span>
                                </label>
                                <label>End:
                                  <input type="number" className="trim-input" min={0} max={dsVideoDuration} step={0.1}
                                    value={dsTrimEnd} onChange={e => {
                                      const v = parseFloat(e.target.value) || 0;
                                      setDsTrimEnd(Math.min(dsVideoDuration, Math.max(v, dsTrimStart + 0.1)));
                                    }} />
                                  <span className="trim-unit">s</span>
                                </label>
                                <div className="trim-selection-info">
                                  {formatTime(dsTrimStart)} - {formatTime(dsTrimEnd)} ({formatTime(dsTrimEnd - dsTrimStart)} selected)
                                </div>
                              </div>
                              <div className="trim-btn-row">
                                <button className="trim-btn" onClick={handleDsTrim} disabled={dsTrimming}>
                                  {dsTrimming ? 'Trimming...' : 'Apply Trim'}
                                </button>
                              </div>
                            </div>
                          </>
                        )}
                      </div>
                    )}

                    {/* Generate + Queue buttons */}
                    <div className="wf-btn-row" style={{ marginTop: 16 }}>
                      <button className="btn generate-btn-green"
                        disabled={!dsCharImagePath || !studioRefVideoPath}
                        onClick={studioHandleChangeCharacter} style={{ flex: 1 }}>
                        {t('studioCreateVideo')}
                      </button>
                      <button className="btn secondary" onClick={handleDsQueueAdd} style={{ flex: 1 }}>
                        {t('wfAddToQueue')}
                      </button>
                    </div>

                    {/* Status */}
                    {studioStatus && (
                      <div className="status-box" style={{ marginTop: 16 }}>
                        <p>{studioStatus}</p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Column - YouTube Settings & Output */}
                <div className="column">
                  {/* YouTube Shorts Settings */}
                  <div className="card yt-meta-card">
                    <h3>{t('studioYtSettings')}</h3>
                    <div className="yt-meta-fields">
                      <label className="yt-meta-label">{t('studioYtChannel')}
                        <input type="text" className="yt-meta-input"
                          value={studioYtChannel}
                          onChange={e => setStudioYtChannel(e.target.value)} />
                      </label>
                      <label className="yt-meta-label">{t('studioYtChannelName')}
                        <input type="text" className="yt-meta-input"
                          value={studioYtChannelName}
                          onChange={e => setStudioYtChannelName(e.target.value)} />
                      </label>
                      <label className="yt-meta-label">{t('ytTitle')}
                        <input type="text" className="yt-meta-input"
                          value={studioYtTitle}
                          onChange={e => setStudioYtTitle(e.target.value)} />
                      </label>
                      <label className="yt-meta-label">{t('ytDescription')}
                        <textarea className="yt-meta-textarea" rows={2}
                          value={studioYtDescription}
                          onChange={e => setStudioYtDescription(e.target.value)} />
                      </label>
                      <label className="yt-meta-label">{t('ytHashtags')}
                        <input type="text" className="yt-meta-input"
                          value={studioYtHashtags}
                          onChange={e => setStudioYtHashtags(e.target.value)} />
                      </label>
                    </div>
                  </div>

                  {/* Output Video & YouTube Upload */}
                  <div className="card" style={{ marginTop: 16 }}>
                    <h3>{t('output')}</h3>
                    {studioIsGenerating && (
                      <div className="progress-container">
                        <div className="progress-bar">
                          <div className="progress-fill" style={{ width: `${studioProgress}%` }} />
                        </div>
                        <span className="progress-text">{studioProgress}%</span>
                      </div>
                    )}
                    {studioStatus && <p className="status-msg">{studioStatus}</p>}
                    {studioOutputVideo && !studioIsGenerating && (
                      <div className="output-container">
                        <video controls src={studioOutputVideo} style={{ width: '100%', borderRadius: 8 }} />
                        <div style={{ marginTop: 12 }}>
                          {studioYtUploadStatus === 'success' ? (
                            <div style={{ textAlign: 'center', padding: 8, background: 'var(--bg-secondary)', borderRadius: 8 }}>
                              <p style={{ color: 'var(--success)', fontWeight: 600, marginBottom: 6 }}>{t('galleryYtSuccess')}</p>
                              <a href={studioYtUploadResult} target="_blank" rel="noopener noreferrer"
                                style={{ color: 'var(--primary)', wordBreak: 'break-all' }}>{studioYtUploadResult}</a>
                            </div>
                          ) : studioYtUploadStatus === 'error' ? (
                            <div style={{ textAlign: 'center', padding: 8, background: 'var(--bg-secondary)', borderRadius: 8 }}>
                              <p style={{ color: 'var(--danger)', marginBottom: 6 }}>{studioYtUploadResult}</p>
                              <button className="btn btn-youtube-dl" onClick={handleStudioYtUpload}>
                                {t('studioYtUploadBtn')}
                              </button>
                            </div>
                          ) : (
                            <button className="btn btn-youtube-dl" style={{ width: '100%' }}
                              disabled={studioYtUploadStatus === 'uploading'}
                              onClick={handleStudioYtUpload}>
                              {studioYtUploadStatus === 'uploading' ? t('studioYtUploading') : t('studioYtUploadBtn')}
                            </button>
                          )}
                        </div>
                        <a href={studioOutputVideo} download className="btn secondary" style={{ marginTop: 8, display: 'inline-block' }}>
                          {t('download')}
                        </a>
                      </div>
                    )}
                    {!studioOutputVideo && !studioIsGenerating && (
                      <p style={{ color: '#888', textAlign: 'center', padding: 40 }}>
                        {t('noOutputYet')}
                      </p>
                    )}
                  </div>

                  {/* Queue Panel */}
                  {(() => {
                    const queue = wfQueue['change_character'];
                    const items = queue?.items || [];
                    const isProcessing = queue?.isProcessing || false;
                    const pendingCount = items.filter(i => i.status === 'pending').length;
                    return (
                      <div className="card queue-card" style={{ marginTop: 16 }}>
                        <h3>{t('wfQueue')} {items.length > 0 && `(${items.length})`}</h3>
                        {items.length === 0 && (
                          <p className="queue-empty">{t('wfQueueEmpty')}</p>
                        )}
                        <div className="queue-list">
                          {items.map(item => (
                            <div key={item.id} className={`queue-item queue-item--${item.status}${(item.previews?.ref_image || item.previews?.ref_video || item.previews?.bg_image) ? ' queue-item--rich' : ''}`}>
                              <div className="queue-item-top">
                                {(item.previews?.ref_image || item.previews?.ref_video || item.previews?.bg_image) && (
                                  <div className="queue-item-thumbs">
                                    {item.previews?.ref_image && (
                                      <img src={item.previews.ref_image} alt="avatar" className="queue-thumb queue-thumb--avatar" />
                                    )}
                                    {item.previews?.ref_video && (
                                      <video src={item.previews.ref_video} className="queue-thumb queue-thumb--video" muted preload="metadata" />
                                    )}
                                    {item.previews?.bg_image && (
                                      <img src={item.previews.bg_image} alt="bg" className="queue-thumb queue-thumb--bg" />
                                    )}
                                  </div>
                                )}
                                <div className="queue-item-info">
                                  <div className="queue-item-title-row">
                                    <span className="queue-item-status">
                                      {item.status === 'completed' ? '\u2705' : item.status === 'running' ? '\u23f3' : item.status === 'failed' ? '\u274c' : '\u23f8'}
                                    </span>
                                    <span className="queue-item-label" title={item.ytTitle || item.inputs?.prompt || ''}>
                                      {item.ytTitle || item.label}
                                    </span>
                                    <button className="queue-item-remove" onClick={() => handleWfQueueRemove('change_character', item.id)}>&times;</button>
                                  </div>
                                  {item.ytTitle && item.label && (
                                    <div className="queue-item-sublabel">{item.label}</div>
                                  )}
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
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                        {items.length > 0 && (
                          <div className="queue-actions">
                            <button className="btn" onClick={() => handleWfQueueStart('change_character')}
                              disabled={isProcessing || pendingCount === 0}>
                              {isProcessing ? t('wfQueueRunning') : t('wfStartQueue')}
                            </button>
                            <button className="btn secondary" onClick={() => handleWfQueueClear('change_character')}
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
          )}

          
          {/* ============ GALLERY ============ */}
          {activeMenu === 'gallery' && (
            <div className="page-content">
              <div className="card">
                <div className="gallery-header">
                  <h3>{t('galleryTitle')}</h3>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    {galleryUploading && (
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-light)' }}>
                        {galleryUploadProgress}%
                      </span>
                    )}
                    <input
                      ref={galleryUploadRef}
                      type="file"
                      accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm"
                      multiple
                      onChange={handleGalleryUpload}
                      style={{ display: 'none' }}
                    />
                    <button
                      className="btn primary"
                      onClick={() => galleryUploadRef.current?.click()}
                      disabled={galleryUploading}
                    >
                      {galleryUploading ? `${t('galleryUploading')} ${galleryUploadProgress}%` : t('galleryUploadBtn')}
                    </button>
                    <button className="btn secondary" onClick={fetchGallery} disabled={galleryLoading}>{galleryLoading ? '...' : t('galleryRefresh')}</button>
                  </div>
                </div>

                <div className="sub-tabs">
                  <button className={galleryTab === 'images' ? 'active' : ''} onClick={() => setGalleryTab('images')}>
                    {t('galleryImages')} ({galleryImages.length})
                  </button>
                  <button className={galleryTab === 'videos' ? 'active' : ''} onClick={() => setGalleryTab('videos')}>
                    {t('galleryVideos')} ({videos.length})
                  </button>
                  <button className={galleryTab === 'stages' ? 'active' : ''} onClick={() => { setGalleryTab('stages'); loadStudioStages(); }}>
                    {t('galleryStages')} ({studioStages.length})
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
                            <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                              {avatarPreparing[img.filename] ? (
                                <button className="btn small" disabled style={{ flex: 1, opacity: 0.7 }}>
                                  {avatarPreparing[img.filename].message || `Preparing... ${avatarPreparing[img.filename].progress}%`}
                                </button>
                              ) : (
                                <button className="btn primary small" onClick={() => handleRegisterAsAvatar(img)}>Avatar</button>
                              )}
                              <button className="btn small" onClick={() => handleRegisterAsStage(img)}>Stage</button>
                              <button className="btn danger small" onClick={() => handleDeleteOutput(img.filename)}>{t('galleryDelete')}</button>
                            </div>
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
                            <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                              <button className="btn primary small" onClick={() => handleYtUploadOpen(video.filename)}>{t('galleryYtUpload')}</button>
                              <button className="btn danger small" onClick={() => handleDeleteOutput(video.filename)}>{t('galleryDelete')}</button>
                            </div>
                          </div>
                          {ytUploadTarget === video.filename && (
                            <div className="yt-upload-form" style={{ padding: '8px 10px', borderTop: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
                              {ytUploadStatus === 'success' ? (
                                <div style={{ textAlign: 'center' }}>
                                  <p style={{ color: 'var(--success)', fontWeight: 600, marginBottom: 6 }}>{t('galleryYtSuccess')}</p>
                                  <a href={ytUploadResult} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)', wordBreak: 'break-all' }}>{ytUploadResult}</a>
                                  <button className="btn secondary small" style={{ marginTop: 8, width: '100%' }} onClick={handleYtUploadClose}>{t('galleryYtCancel')}</button>
                                </div>
                              ) : ytUploadStatus === 'error' ? (
                                <div style={{ textAlign: 'center' }}>
                                  <p style={{ color: 'var(--danger)', marginBottom: 6 }}>{ytUploadResult}</p>
                                  <button className="btn secondary small" style={{ width: '100%' }} onClick={handleYtUploadClose}>{t('galleryYtCancel')}</button>
                                </div>
                              ) : (
                                <>
                                  <input type="text" placeholder={t('galleryYtTitle')} value={ytUploadForm.title}
                                    onChange={e => setYtUploadForm(f => ({ ...f, title: e.target.value }))}
                                    style={{ width: '100%', marginBottom: 4, padding: '4px 6px', fontSize: 12 }} />
                                  <input type="text" placeholder={t('galleryYtDesc')} value={ytUploadForm.description}
                                    onChange={e => setYtUploadForm(f => ({ ...f, description: e.target.value }))}
                                    style={{ width: '100%', marginBottom: 4, padding: '4px 6px', fontSize: 12 }} />
                                  <input type="text" placeholder={t('galleryYtHashtags') + ' (#AI #dance)'} value={ytUploadForm.hashtags}
                                    onChange={e => setYtUploadForm(f => ({ ...f, hashtags: e.target.value }))}
                                    style={{ width: '100%', marginBottom: 6, padding: '4px 6px', fontSize: 12 }} />
                                  <div style={{ display: 'flex', gap: 6 }}>
                                    <button className="btn primary small" style={{ flex: 1 }} onClick={handleYtUploadSubmit}
                                      disabled={ytUploadStatus === 'uploading'}>
                                      {ytUploadStatus === 'uploading' ? t('galleryYtUploading') : t('galleryYtUpload')}
                                    </button>
                                    <button className="btn secondary small" style={{ flex: 1 }} onClick={handleYtUploadClose}
                                      disabled={ytUploadStatus === 'uploading'}>{t('galleryYtCancel')}</button>
                                  </div>
                                </>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )
                )}

                {galleryTab === 'stages' && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
                      <label className="btn primary small" style={{ cursor: 'pointer' }}>
                        <input type="file" accept="image/*" onChange={handleGalleryStageUpload} style={{ display: 'none' }} />
                        + {t('studioStageUpload')}
                      </label>
                    </div>
                    {studioStages.length === 0 ? (
                      <div className="gallery-empty"><p>{t('galleryEmpty')}</p></div>
                    ) : (
                      <div className="gallery-grid">
                        {studioStages.map(stage => (
                          <div key={stage.filename} className="gallery-item">
                            <img src={stage.url} alt={stage.filename} className="gallery-item-img" />
                            <div className="gallery-item-info">
                              <span className="gallery-item-name" title={stage.filename}>{stage.filename}</span>
                              <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                                <button className="btn danger small" onClick={() => handleStudioStageDelete(stage.filename)}>{t('galleryDelete')}</button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
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

      {/* Avatar Image Popup Viewer */}
      {avatarPopup && (() => {
        const popupImages = avatarPopup.source === 'danceshorts' ? dsAvatarImages
          : (workflowStates[avatarPopup.wfId]?.avatarImages?.[avatarPopup.group] || []);
        const showNav = popupImages.length > 1;
        return (
        <div className="avatar-popup-overlay" onClick={() => setAvatarPopup(null)}>
          {showNav && <button className="avatar-popup-nav avatar-popup-nav-left" onClick={(e) => { e.stopPropagation(); avatarPopupNavigate('left'); }}>&#10094;</button>}
          <div className="avatar-popup" onClick={e => e.stopPropagation()}>
            <div className="avatar-popup-header">
              <span className="avatar-popup-name">{avatarPopup.filename?.replace(/\.[^.]+$/, '') || ''}</span>
              <button className="avatar-popup-close" onClick={() => setAvatarPopup(null)}>&times;</button>
            </div>
            <div className="avatar-popup-body">
              <img src={avatarPopup.url} alt={avatarPopup.filename || ''} />
            </div>
            {avatarPopup.img && (
              <div className="avatar-popup-footer">
                <button className="btn cancel-btn" onClick={async () => {
                  if (!window.confirm(`Delete "${avatarPopup.filename}" from ${avatarPopup.group}?`)) return;
                  if (avatarPopup.source === 'danceshorts') {
                    await deleteAvatarImage(avatarPopup.group, avatarPopup.img.filename);
                    await loadDsAvatarImages(avatarPopup.group);
                    if (dsCharImagePath === avatarPopup.img.path) {
                      setDsCharImagePath('');
                      setDsCharImagePreview(null);
                    }
                  } else {
                    await handleWfAvatarDelete(avatarPopup.wfId, avatarPopup.group, avatarPopup.img);
                  }
                  setAvatarPopup(null);
                }}>
                  {t('galleryDelete')}
                </button>
              </div>
            )}
          </div>
          {showNav && <button className="avatar-popup-nav avatar-popup-nav-right" onClick={(e) => { e.stopPropagation(); avatarPopupNavigate('right'); }}>&#10095;</button>}
        </div>
        );
      })()}
    </div>
  );
}

export default App;
