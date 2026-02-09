import { useState, useEffect, useRef, useCallback } from 'react';
import {
  uploadImage,
  uploadAudio,
  uploadVideo,
  startGeneration,
  getTaskStatus,
  extractAudio,
  separateVocals,
  getConfig,
} from './api';
import './App.css';

// Translations
const translations = {
  en: {
    title: 'WanAvatar',
    subtitle: 'Talking Head Video Generation powered by Wan2.2-S2V-14B',
    tabGenerate: 'Generate Video',
    tabExtract: 'Extract Audio',
    tabSeparate: 'Vocal Separation',
    imageLabel: 'Reference Image',
    audioLabel: 'Driving Audio',
    promptLabel: 'Prompt',
    promptDefault: 'A person speaking naturally with subtle expressions, minimal head movement, simple blinking, neutral background',
    negPromptLabel: 'Negative Prompt',
    negPromptDefault: 'blurry, low quality, distorted face, unnatural movements, artifacts, text, watermark',
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
    videoInput: 'Upload Video',
    extractBtn: 'Extract Audio',
    audioOutput: 'Extracted Audio',
    audioInput: 'Upload Audio',
    separateBtn: 'Separate Vocals',
    vocalsOutput: 'Separated Vocals',
    dropImage: 'Drop image here or click to upload',
    dropAudio: 'Drop audio here or click to upload',
    dropVideo: 'Drop video here or click to upload',
  },
  ko: {
    title: 'WanAvatar',
    subtitle: 'Wan2.2-S2V-14B 기반 토킹 헤드 비디오 생성',
    tabGenerate: '비디오 생성',
    tabExtract: '오디오 추출',
    tabSeparate: '보컬 분리',
    imageLabel: '참조 이미지',
    audioLabel: '구동 오디오',
    promptLabel: '프롬프트',
    promptDefault: '자연스럽게 말하는 사람, 미세한 표정, 최소한의 머리 움직임, 단순한 눈 깜빡임, 중립적 배경',
    negPromptLabel: '네거티브 프롬프트',
    negPromptDefault: '흐림, 저화질, 왜곡된 얼굴, 부자연스러운 움직임, 아티팩트, 텍스트, 워터마크',
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
  },
  zh: {
    title: 'WanAvatar',
    subtitle: '基于 Wan2.2-S2V-14B 的说话人头视频生成',
    tabGenerate: '生成视频',
    tabExtract: '提取音频',
    tabSeparate: '人声分离',
    imageLabel: '参考图片',
    audioLabel: '驱动音频',
    promptLabel: '提示词',
    promptDefault: '一个人自然说话，细微表情，头部动作轻微，简单眨眼，中性背景',
    negPromptLabel: '负面提示词',
    negPromptDefault: '模糊，低质量，扭曲的脸，不自然的动作，伪影，文字，水印',
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
  },
};

function App() {
  const [lang, setLang] = useState('en');
  const [activeTab, setActiveTab] = useState('generate');
  const [config, setConfig] = useState(null);

  // Generate tab state
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
  const [steps, setSteps] = useState(15);
  const [guidance, setGuidance] = useState(4.5);
  const [frames, setFrames] = useState(80);
  const [seed, setSeed] = useState(-1);
  const [offload, setOffload] = useState(false);
  const [useTeacache, setUseTeacache] = useState(true);
  const [teacacheThresh, setTeacacheThresh] = useState(0.3);
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

  const t = useCallback((key) => translations[lang][key] || key, [lang]);

  // Load config on mount
  useEffect(() => {
    getConfig().then(res => setConfig(res.data)).catch(console.error);
  }, []);

  // Update prompts when language changes
  useEffect(() => {
    setPrompt(translations[lang].promptDefault);
    setNegPrompt(translations[lang].negPromptDefault);
  }, [lang]);

  // File upload handlers
  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));

    try {
      const result = await uploadImage(file);
      setImagePath(result.path);

      // Store image dimensions
      if (result.width && result.height) {
        setImageDimensions({ width: result.width, height: result.height });

        // Auto-set resolution if enabled
        if (autoResolution) {
          const autoRes = `${result.height}*${result.width}`;
          setResolution(autoRes);
        }
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

  const handleExtractVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setExtractVideoFile(file);

    try {
      const result = await uploadVideo(file);
      setExtractVideoPath(result.path);
    } catch (err) {
      setExtractStatus(`Error uploading video: ${err.message}`);
    }
  };

  const handleSeparateAudioUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setSeparateAudioFile(file);

    try {
      const result = await uploadAudio(file);
      setSeparateAudioPath(result.path);
    } catch (err) {
      setSeparateStatus(`Error uploading audio: ${err.message}`);
    }
  };

  // Generation handler
  const handleGenerate = async () => {
    if (!imagePath || !audioPath) {
      setStatus('Please upload both image and audio');
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setStatus('Starting generation...');
    setOutputVideo(null);

    try {
      const { task_id } = await startGeneration({
        image_path: imagePath,
        audio_path: audioPath,
        prompt,
        negative_prompt: negPrompt,
        resolution,
        num_clips: numClips,
        inference_steps: steps,
        guidance_scale: guidance,
        infer_frames: frames,
        seed,
        offload_model: offload,
        use_teacache: useTeacache,
        teacache_thresh: teacacheThresh,
      });

      // Poll for status
      const pollInterval = setInterval(async () => {
        try {
          const taskStatus = await getTaskStatus(task_id);
          setProgress(taskStatus.progress * 100);
          setStatus(taskStatus.message);

          if (taskStatus.status === 'completed') {
            clearInterval(pollInterval);
            setIsGenerating(false);
            setOutputVideo(taskStatus.output_path);
            setOutputSeed(taskStatus.seed?.toString() || '');
          } else if (taskStatus.status === 'failed') {
            clearInterval(pollInterval);
            setIsGenerating(false);
            setStatus(`Error: ${taskStatus.message}`);
          }
        } catch (err) {
          clearInterval(pollInterval);
          setIsGenerating(false);
          setStatus(`Error: ${err.message}`);
        }
      }, 2000);
    } catch (err) {
      setIsGenerating(false);
      setStatus(`Error: ${err.message}`);
    }
  };

  // Extract audio handler
  const handleExtract = async () => {
    if (!extractVideoPath) {
      setExtractStatus('Please upload a video');
      return;
    }

    setIsExtracting(true);
    setExtractStatus('Extracting audio...');
    setExtractedAudio(null);

    try {
      const result = await extractAudio(extractVideoPath);
      setExtractedAudio(result.url);
      setExtractStatus('Audio extracted successfully!');
    } catch (err) {
      setExtractStatus(`Error: ${err.message}`);
    } finally {
      setIsExtracting(false);
    }
  };

  // Separate vocals handler
  const handleSeparate = async () => {
    if (!separateAudioPath) {
      setSeparateStatus('Please upload audio');
      return;
    }

    setIsSeparating(true);
    setSeparateStatus('Separating vocals...');
    setSeparatedVocals(null);

    try {
      const result = await separateVocals(separateAudioPath);
      setSeparatedVocals(result.url);
      setSeparateStatus('Vocals separated successfully!');
    } catch (err) {
      setSeparateStatus(`Error: ${err.message}`);
    } finally {
      setIsSeparating(false);
    }
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1 className="title">{t('title')}</h1>
        <p className="subtitle">{t('subtitle')}</p>
      </header>

      {/* Language Selector */}
      <div className="language-selector">
        <button
          className={lang === 'en' ? 'active' : ''}
          onClick={() => setLang('en')}
        >
          English
        </button>
        <button
          className={lang === 'ko' ? 'active' : ''}
          onClick={() => setLang('ko')}
        >
          한국어
        </button>
        <button
          className={lang === 'zh' ? 'active' : ''}
          onClick={() => setLang('zh')}
        >
          中文
        </button>
      </div>

      {/* Tabs */}
      <nav className="tabs">
        <button
          className={activeTab === 'generate' ? 'active' : ''}
          onClick={() => setActiveTab('generate')}
        >
          {t('tabGenerate')}
        </button>
        <button
          className={activeTab === 'extract' ? 'active' : ''}
          onClick={() => setActiveTab('extract')}
        >
          {t('tabExtract')}
        </button>
        <button
          className={activeTab === 'separate' ? 'active' : ''}
          onClick={() => setActiveTab('separate')}
        >
          {t('tabSeparate')}
        </button>
      </nav>

      {/* Generate Tab */}
      {activeTab === 'generate' && (
        <div className="tab-content generate-tab">
          <div className="two-column">
            {/* Left Column - Inputs */}
            <div className="column">
              <div className="card">
                <h3>Input</h3>

                {/* Image Upload */}
                <div className="form-group">
                  <label>{t('imageLabel')}</label>
                  <div className="file-upload">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      id="image-upload"
                    />
                    <label htmlFor="image-upload" className="upload-area">
                      {imagePreview ? (
                        <img src={imagePreview} alt="Preview" className="preview-image" />
                      ) : (
                        <span>{t('dropImage')}</span>
                      )}
                    </label>
                  </div>
                </div>

                {/* Audio Upload */}
                <div className="form-group">
                  <label>{t('audioLabel')}</label>
                  <div className="file-upload">
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={handleAudioUpload}
                      id="audio-upload"
                    />
                    <label htmlFor="audio-upload" className="upload-area small">
                      {audioFile ? (
                        <span>{audioFile.name}</span>
                      ) : (
                        <span>{t('dropAudio')}</span>
                      )}
                    </label>
                  </div>
                </div>
              </div>

              {/* Settings */}
              <div className="card">
                <h3>Settings</h3>

                <div className="form-group">
                  <label>{t('promptLabel')}</label>
                  <textarea
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={3}
                  />
                </div>

                <div className="form-group">
                  <label>{t('negPromptLabel')}</label>
                  <textarea
                    value={negPrompt}
                    onChange={(e) => setNegPrompt(e.target.value)}
                    rows={2}
                  />
                </div>

                <div className="form-group checkbox">
                  <label>
                    <input
                      type="checkbox"
                      checked={autoResolution}
                      onChange={(e) => {
                        setAutoResolution(e.target.checked);
                        if (e.target.checked && imageDimensions) {
                          setResolution(`${imageDimensions.height}*${imageDimensions.width}`);
                        }
                      }}
                    />
                    {t('autoResolution')}
                  </label>
                  {imageDimensions && (
                    <span className="image-size-info">
                      {t('imageSize')}: {imageDimensions.width} x {imageDimensions.height}
                    </span>
                  )}
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>{t('resolutionLabel')}: {resolution}</label>
                    {autoResolution ? (
                      <input
                        type="text"
                        value={resolution}
                        onChange={(e) => setResolution(e.target.value)}
                        placeholder="height*width"
                      />
                    ) : (
                      <select value={resolution} onChange={(e) => setResolution(e.target.value)}>
                        {config?.resolutions?.map((r) => (
                          <option key={r} value={r}>{r}</option>
                        )) || (
                          <>
                            <option value="720*1280">720*1280</option>
                            <option value="1280*720">1280*720</option>
                            <option value="480*832">480*832</option>
                          </>
                        )}
                      </select>
                    )}
                  </div>
                  <div className="form-group">
                    <label>{t('clipsLabel')}</label>
                    <input
                      type="number"
                      min={0}
                      max={10}
                      value={numClips}
                      onChange={(e) => setNumClips(parseInt(e.target.value))}
                    />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>{t('stepsLabel')}: {steps}</label>
                    <input
                      type="range"
                      min={5}
                      max={50}
                      step={1}
                      value={steps}
                      onChange={(e) => setSteps(parseInt(e.target.value))}
                    />
                  </div>
                  <div className="form-group">
                    <label>{t('guidanceLabel')}: {guidance}</label>
                    <input
                      type="range"
                      min={1}
                      max={10}
                      step={0.5}
                      value={guidance}
                      onChange={(e) => setGuidance(parseFloat(e.target.value))}
                    />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>{t('framesLabel')}: {frames}</label>
                    <input
                      type="range"
                      min={48}
                      max={120}
                      step={4}
                      value={frames}
                      onChange={(e) => setFrames(parseInt(e.target.value))}
                    />
                  </div>
                  <div className="form-group">
                    <label>{t('seedLabel')}</label>
                    <input
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(parseInt(e.target.value))}
                    />
                  </div>
                </div>

                <div className="form-group checkbox">
                  <label>
                    <input
                      type="checkbox"
                      checked={offload}
                      onChange={(e) => setOffload(e.target.checked)}
                    />
                    {t('offloadLabel')}
                  </label>
                </div>

                <div className="form-group checkbox">
                  <label>
                    <input
                      type="checkbox"
                      checked={useTeacache}
                      onChange={(e) => setUseTeacache(e.target.checked)}
                    />
                    {t('teacacheLabel')}
                  </label>
                </div>

                {useTeacache && (
                  <div className="form-group">
                    <label>{t('teacacheThreshLabel')}: {teacacheThresh}</label>
                    <input
                      type="range"
                      min={0.05}
                      max={1.0}
                      step={0.05}
                      value={teacacheThresh}
                      onChange={(e) => setTeacacheThresh(parseFloat(e.target.value))}
                    />
                  </div>
                )}

                <button
                  className="btn primary"
                  onClick={handleGenerate}
                  disabled={isGenerating || !imagePath || !audioPath}
                >
                  {isGenerating ? t('generating') : t('generateBtn')}
                </button>
              </div>
            </div>

            {/* Right Column - Output */}
            <div className="column">
              <div className="card">
                <h3>Output</h3>

                {/* Progress */}
                {isGenerating && (
                  <div className="progress-container">
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                    <span className="progress-text">{Math.round(progress)}%</span>
                  </div>
                )}

                {/* Video Output */}
                <div className="video-container">
                  {outputVideo ? (
                    <video controls src={outputVideo} />
                  ) : (
                    <div className="placeholder">
                      <span>{t('videoOutput')}</span>
                    </div>
                  )}
                </div>

                {/* Seed */}
                {outputSeed && (
                  <div className="form-group">
                    <label>{t('seedOutput')}</label>
                    <input type="text" value={outputSeed} readOnly />
                  </div>
                )}

                {/* Status */}
                <div className="status-box">
                  <label>{t('status')}</label>
                  <p>{status || 'Ready'}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Extract Tab */}
      {activeTab === 'extract' && (
        <div className="tab-content extract-tab">
          <div className="two-column">
            <div className="column">
              <div className="card">
                <h3>{t('videoInput')}</h3>
                <div className="file-upload">
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleExtractVideoUpload}
                    id="extract-video-upload"
                  />
                  <label htmlFor="extract-video-upload" className="upload-area">
                    {extractVideoFile ? (
                      <span>{extractVideoFile.name}</span>
                    ) : (
                      <span>{t('dropVideo')}</span>
                    )}
                  </label>
                </div>
                <button
                  className="btn primary"
                  onClick={handleExtract}
                  disabled={isExtracting || !extractVideoPath}
                >
                  {isExtracting ? '...' : t('extractBtn')}
                </button>
              </div>
            </div>
            <div className="column">
              <div className="card">
                <h3>{t('audioOutput')}</h3>
                {extractedAudio ? (
                  <audio controls src={extractedAudio} />
                ) : (
                  <div className="placeholder small">
                    <span>{t('audioOutput')}</span>
                  </div>
                )}
                <div className="status-box">
                  <label>{t('status')}</label>
                  <p>{extractStatus || 'Ready'}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Separate Tab */}
      {activeTab === 'separate' && (
        <div className="tab-content separate-tab">
          <div className="two-column">
            <div className="column">
              <div className="card">
                <h3>{t('audioInput')}</h3>
                <div className="file-upload">
                  <input
                    type="file"
                    accept="audio/*"
                    onChange={handleSeparateAudioUpload}
                    id="separate-audio-upload"
                  />
                  <label htmlFor="separate-audio-upload" className="upload-area">
                    {separateAudioFile ? (
                      <span>{separateAudioFile.name}</span>
                    ) : (
                      <span>{t('dropAudio')}</span>
                    )}
                  </label>
                </div>
                <button
                  className="btn primary"
                  onClick={handleSeparate}
                  disabled={isSeparating || !separateAudioPath}
                >
                  {isSeparating ? '...' : t('separateBtn')}
                </button>
              </div>
            </div>
            <div className="column">
              <div className="card">
                <h3>{t('vocalsOutput')}</h3>
                {separatedVocals ? (
                  <audio controls src={separatedVocals} />
                ) : (
                  <div className="placeholder small">
                    <span>{t('vocalsOutput')}</span>
                  </div>
                )}
                <div className="status-box">
                  <label>{t('status')}</label>
                  <p>{separateStatus || 'Ready'}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <p>Powered by Wan2.2-S2V-14B | WanAvatar</p>
      </footer>
    </div>
  );
}

export default App;
