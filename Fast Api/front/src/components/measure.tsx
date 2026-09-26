import React, { useState, useRef } from 'react';

const JETSON_URL = import.meta.env.VITE_JETSON_URL as string | undefined;

type InputMode = 'file' | 'jetson';
type ScaleMode = 'board' | 'hardware';
type FingerGroup = 'thumb' | 'other';

type MeasureResult = {
  length_mm?: number | null;
  width_mm?: number | null;
  shape?: string | null;
  contours?: unknown;
  preview?: string;
};

type MeasurePayload = {
  front: MeasureResult[];
  side: MeasureResult[] | null;
};

type ShotView = {
  label: string;
  preview: string;
  results: MeasureResult[] | null;
  error: string | null;
  imageSize: { width: number; height: number };
  displaySize: { width: number; height: number };
};

function errorText(data: { detail?: unknown } | null, fallback: string): string {
  const detail = data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item: { msg?: string }) => item?.msg ?? JSON.stringify(item)).join('\n');
  }
  return fallback;
}

function b64ToFile(b64: string, name: string): File {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new File([bytes], name, { type: 'image/jpeg' });
}

async function postMeasure(
  file: File,
  scale: ScaleMode,
  group: FingerGroup,
  side?: File | null,
): Promise<{ payload: MeasurePayload | null; error: string | null }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('scale', scale);
  formData.append('group', group);
  if (side) formData.append('side', side);
  try {
    const response = await fetch('http://localhost:8000/api/measure', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) return { payload: null, error: errorText(data, '측정에 실패했습니다.') };
    if (!data || !Array.isArray(data.front)) return { payload: null, error: '예상하지 못한 응답입니다.' };
    return {
      payload: { front: data.front, side: Array.isArray(data.side) ? data.side : null },
      error: null,
    };
  } catch {
    return { payload: null, error: '서버에 연결하지 못했습니다.' };
  }
}

function ResultList({ results, scale }: { results: MeasureResult[]; scale: ScaleMode }) {
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', justifyContent: 'center' }}>
      {results.map((res, index) => (
        <div key={index} style={{ margin: '10px 0', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px', width: '160px' }}>
          {res.preview && (
            <img
              src={res.preview}
              alt={`전처리 ${index + 1}`}
              style={{ width: '140px', height: '140px', objectFit: 'contain', background: '#000', display: 'block', margin: '0 auto 8px' }}
            />
          )}
          {scale === 'hardware'
            ? '길이/폭: 하드웨어 실측 미구현'
            : `길이 ${res.length_mm ?? '-'}mm / 폭 ${res.width_mm ? `${res.width_mm}mm` : '측정 불가'}`}
          {res.shape ? ` / ${res.shape}형입니다` : ''}
        </div>
      ))}
    </div>
  );
}

export default function NailMeasurement() {
  const [inputMode, setInputMode] = useState<InputMode>('file');
  const [scaleMode, setScaleMode] = useState<ScaleMode>('board');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [fingerGroup, setFingerGroup] = useState<FingerGroup>('other');
  const [measurementResults, setMeasurementResults] = useState<MeasureResult[] | null>(null);
  const [shots, setShots] = useState<ShotView[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });
  const imageRef = useRef<HTMLImageElement>(null);

  const syncImageSize = () => {
    const img = imageRef.current;
    if (!img) return;
    setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
    setDisplaySize({ width: img.clientWidth, height: img.clientHeight });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setMeasurementResults(null);
      setShots([]);
      setErrorMessage(null);
    }
  };

  const handleSideChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files && e.target.files.length > 0 ? e.target.files[0] : null;
    setSideFile(file);
    setMeasurementResults(null);
    setShots([]);
    setErrorMessage(null);
  };

  const handleSubmit = async () => {
    if (!imageFile) return;
    const measured = await postMeasure(imageFile, scaleMode, fingerGroup, sideFile);
    setShots([]);
    setMeasurementResults(measured.payload?.front ?? null);
    setErrorMessage(measured.error);
    if (measured.payload?.side && sideFile) {
      const emptySize = { width: 0, height: 0 };
      setShots([
        { label: '정면', preview: URL.createObjectURL(imageFile), results: measured.payload.front, error: null, imageSize: emptySize, displaySize: emptySize },
        { label: '측면', preview: URL.createObjectURL(sideFile), results: measured.payload.side, error: null, imageSize: emptySize, displaySize: emptySize },
      ]);
    }
  };

  const handleJetsonShot = async () => {
    if (!JETSON_URL) {
      setErrorMessage('VITE_JETSON_URL이 없습니다.');
      return;
    }
    setErrorMessage(null);
    setMeasurementResults(null);
    try {
      const response = await fetch(`${JETSON_URL}/shot`);
      const data = await response.json();
      if (!response.ok || !data?.front || !data?.side) {
        setShots([]);
        setErrorMessage(errorText(data, 'Jetson 촬영에 실패했습니다.'));
        return;
      }
      const frontFile = b64ToFile(data.front, 'front.jpg');
      const sideFile = b64ToFile(data.side, 'side.jpg');
      const emptySize = { width: 0, height: 0 };
      setShots([
        { label: '정면', preview: URL.createObjectURL(frontFile), results: null, error: null, imageSize: emptySize, displaySize: emptySize },
        { label: '측면', preview: URL.createObjectURL(sideFile), results: null, error: null, imageSize: emptySize, displaySize: emptySize },
      ]);
      const measured = await postMeasure(frontFile, scaleMode, fingerGroup, sideFile);
      if (!measured.payload) {
        setShots((prev) => prev.map((shot) => ({ ...shot, results: null, error: measured.error })));
        return;
      }
      setShots((prev) => prev.map((shot, index) => ({
        ...shot,
        results: index === 0 ? measured.payload!.front : measured.payload!.side,
        error: null,
      })));
    } catch {
      setShots([]);
      setErrorMessage('Jetson에 연결하지 못했습니다.');
    }
  };

  const bindShotSize = (index: number) => (event: React.SyntheticEvent<HTMLImageElement>) => {
    const img = event.currentTarget;
    setShots((prev) => prev.map((shot, i) => i === index ? {
      ...shot,
      imageSize: { width: img.naturalWidth, height: img.naturalHeight },
      displaySize: { width: img.clientWidth, height: img.clientHeight },
    } : shot));
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '50px' }}>
      <h2>손톱 측정</h2>

      <div style={{ display: 'flex', gap: '28px', marginBottom: '16px', fontSize: '14px' }}>
        <div>
          <div style={{ marginBottom: '6px' }}>입력</div>
          <label>
            <input type="radio" name="inputMode" checked={inputMode === 'file'} onChange={() => { setInputMode('file'); setShots([]); setMeasurementResults(null); setErrorMessage(null); }} />
            {' '}사진 파일
          </label>
          <label style={{ marginLeft: '12px' }}>
            <input type="radio" name="inputMode" checked={inputMode === 'jetson'} onChange={() => { setInputMode('jetson'); setShots([]); setMeasurementResults(null); setErrorMessage(null); }} />
            {' '}Jetson
          </label>
        </div>
        <div>
          <div style={{ marginBottom: '6px' }}>실측</div>
          <label>
            <input type="radio" name="scaleMode" checked={scaleMode === 'board'} onChange={() => setScaleMode('board')} />
            {' '}체커보드
          </label>
          <label style={{ marginLeft: '12px' }}>
            <input type="radio" name="scaleMode" checked={scaleMode === 'hardware'} onChange={() => setScaleMode('hardware')} />
            {' '}하드웨어
          </label>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '16px', marginBottom: '12px', fontSize: '14px' }}>
        <div style={{ marginRight: '8px' }}>분류</div>
        <label>
          <input type="radio" name="fingerGroup" checked={fingerGroup === 'thumb'} onChange={() => setFingerGroup('thumb')} />
          {' '}엄지
        </label>
        <label>
          <input type="radio" name="fingerGroup" checked={fingerGroup === 'other'} onChange={() => setFingerGroup('other')} />
          {' '}나머지
        </label>
      </div>

      {inputMode === 'file' && (
        <div style={{ position: 'relative', display: 'inline-block', marginBottom: '20px' }}>
          <label style={{ display: 'block', marginBottom: '8px', fontSize: '14px' }}>
            정면
            <input type="file" accept="image/*" onChange={handleFileChange} style={{ marginLeft: '8px' }} />
          </label>
          <label style={{ display: 'block', marginBottom: '10px', fontSize: '14px' }}>
            측면
            <input type="file" accept="image/*" onChange={handleSideChange} style={{ marginLeft: '8px' }} />
          </label>
          {imagePreview && shots.length === 0 && (
            <div style={{ position: 'relative', display: 'block', width: 'fit-content', lineHeight: 0 }}>
              <img
                ref={imageRef}
                src={imagePreview}
                alt="preview"
                onLoad={syncImageSize}
                style={{ maxWidth: '400px', width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }}
              />
              {Array.isArray(measurementResults) && imageSize.width > 0 && displaySize.width > 0 && (
                <svg
                  width={displaySize.width}
                  height={displaySize.height}
                  viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                  preserveAspectRatio="xMidYMid meet"
                  style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
                >
                  {measurementResults.map((res, index) => {
                    if (!Array.isArray(res.contours)) return null;
                    const d = res.contours
                      .filter((cnt: unknown) => Array.isArray(cnt) && cnt.length > 0)
                      .map((cnt: number[][]) => {
                        const pts = cnt.filter((p) => Array.isArray(p) && p.length >= 2).map((p) => `${p[0]},${p[1]}`);
                        if (pts.length === 0) return '';
                        return `M ${pts.join(' L ')} Z`;
                      })
                      .join(' ');
                    if (!d.trim()) return null;
                    return (
                      <path key={index} d={d} fill="rgba(255, 0, 85, 0.3)" stroke="#ff0055" strokeWidth="2" />
                    );
                  })}
                </svg>
              )}
            </div>
          )}
          <button onClick={handleSubmit} disabled={!imageFile} style={{ marginTop: '12px', padding: '10px 30px', cursor: imageFile ? 'pointer' : 'not-allowed' }}>
            API 전송 및 탐지
          </button>
        </div>
      )}

      {inputMode === 'jetson' && (
        <button onClick={handleJetsonShot} style={{ marginBottom: '20px', padding: '10px 30px' }}>
          Jetson 촬영
        </button>
      )}

      {errorMessage && <p style={{ marginTop: '16px', color: '#c00' }}>{errorMessage}</p>}

      {inputMode === 'file' && shots.length === 0 && Array.isArray(measurementResults) && (
        <div style={{ marginTop: '30px', textAlign: 'center', width: '100%', maxWidth: '720px' }}>
          <h3>측정 결과</h3>
          <ResultList results={measurementResults} scale={scaleMode} />
        </div>
      )}

      {shots.length > 0 && (
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', justifyContent: 'center', marginTop: '12px' }}>
          {shots.map((shot, index) => (
            <div key={shot.label} style={{ textAlign: 'center' }}>
              <h3>{shot.label}</h3>
              <div style={{ position: 'relative', display: 'inline-block', lineHeight: 0 }}>
                <img
                  src={shot.preview}
                  alt={shot.label}
                  onLoad={bindShotSize(index)}
                  style={{ maxWidth: '400px', width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }}
                />
                {Array.isArray(shot.results) && shot.imageSize.width > 0 && shot.displaySize.width > 0 && (
                  <svg
                    width={shot.displaySize.width}
                    height={shot.displaySize.height}
                    viewBox={`0 0 ${shot.imageSize.width} ${shot.imageSize.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
                  >
                    {shot.results.map((res, contourIndex) => {
                      if (!Array.isArray(res.contours)) return null;
                      const d = res.contours
                        .filter((cnt: unknown) => Array.isArray(cnt) && cnt.length > 0)
                        .map((cnt: number[][]) => {
                          const pts = cnt.filter((p) => Array.isArray(p) && p.length >= 2).map((p) => `${p[0]},${p[1]}`);
                          if (pts.length === 0) return '';
                          return `M ${pts.join(' L ')} Z`;
                        })
                        .join(' ');
                      if (!d.trim()) return null;
                      return <path key={contourIndex} d={d} fill="rgba(255, 0, 85, 0.3)" stroke="#ff0055" strokeWidth="2" />;
                    })}
                  </svg>
                )}
              </div>
              {shot.error && <p style={{ color: '#c00' }}>{shot.error}</p>}
              {Array.isArray(shot.results) && <ResultList results={shot.results} scale={scaleMode} />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
