import React, { useState, useRef } from 'react';

const JETSON_URL = import.meta.env.VITE_JETSON_URL;

type MeasureResult = {
  length_mm?: number;
  width_mm?: number | null;
  shape?: string;
  metric?: string;
  shape_score?: number | null;
  contours?: unknown;
  preview?: string;
};

type Box = { width: number; height: number };

type ShotView = {
  preview: string;
  results: MeasureResult[] | null;
  error: string | null;
  imageSize: Box;
  displaySize: Box;
};

const emptyBox = { width: 0, height: 0 };

function errorText(data: { detail?: unknown } | null): string {
  const detail = data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item: { msg?: string }) => item?.msg ?? JSON.stringify(item)).join('\n');
  }
  return '측정에 실패했습니다.';
}

function b64ToFile(b64: string, name: string): File {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return new File([bytes], name, { type: 'image/jpeg' });
}

async function postMeasure(file: File, metric: string): Promise<{ results: MeasureResult[] | null; error: string | null }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('metric', metric);
  try {
    const response = await fetch('http://localhost:8000/api/measure', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) return { results: null, error: errorText(data) };
    if (!Array.isArray(data)) return { results: null, error: '예상하지 못한 응답입니다.' };
    return { results: data, error: null };
  } catch {
    return { results: null, error: '서버에 연결하지 못했습니다.' };
  }
}

function ContourOverlay({
  results,
  imageSize,
  displaySize,
}: {
  results: MeasureResult[] | null;
  imageSize: Box;
  displaySize: Box;
}) {
  if (!Array.isArray(results) || imageSize.width <= 0 || displaySize.width <= 0) return null;
  return (
    <svg
      width={displaySize.width}
      height={displaySize.height}
      viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}
    >
      {results.map((res, index) => {
        if (!Array.isArray(res.contours)) return null;
        const d = res.contours
          .filter((cnt: unknown) => Array.isArray(cnt) && cnt.length > 0)
          .map((cnt: number[][]) => {
            const pts = cnt
              .filter((p) => Array.isArray(p) && p.length >= 2)
              .map((p) => `${p[0]},${p[1]}`);
            if (pts.length === 0) return '';
            return `M ${pts.join(' L ')} Z`;
          })
          .join(' ');
        if (!d.trim()) return null;
        return (
          <path
            key={index}
            d={d}
            fill="rgba(255, 0, 85, 0.3)"
            stroke="#ff0055"
            strokeWidth="2"
          />
        );
      })}
    </svg>
  );
}

function ResultList({ results }: { results: MeasureResult[] | null }) {
  if (!Array.isArray(results)) return null;
  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: '8px 0 0' }}>
      {results.map((res, index) => (
        <li key={index} style={{ margin: '10px 0', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px', width: '160px' }}>
          {res.preview && (
            <img
              src={res.preview}
              alt={`전처리 ${index + 1}`}
              style={{ width: '140px', height: '140px', objectFit: 'contain', background: '#000', display: 'block', margin: '0 auto 8px' }}
            />
          )}
          {res.shape ? `쉐입 ${res.shape}` : '쉐입 없음'}
          {res.metric ? ` / ${res.metric === 'xor' ? 'XOR' : 'Chamfer'}` : ''}
          {res.shape_score != null ? ` (${res.shape_score})` : ''}
        </li>
      ))}
    </ul>
  );
}

export default function NailMeasurement() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [metric, setMetric] = useState<'chamfer' | 'xor'>('chamfer');
  const [measurementResults, setMeasurementResults] = useState<MeasureResult[] | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<Box>({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState<Box>({ width: 0, height: 0 });
  const [frontShot, setFrontShot] = useState<ShotView | null>(null);
  const [sideShot, setSideShot] = useState<ShotView | null>(null);
  const [captureError, setCaptureError] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
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
      setErrorMessage(null);
    }
  };

  const handleSubmit = async () => {
    if (!imageFile) return;
    const measured = await postMeasure(imageFile, metric);
    setMeasurementResults(measured.results);
    setErrorMessage(measured.error);
  };

  const bindShotSize = (
    setter: React.Dispatch<React.SetStateAction<ShotView | null>>,
  ) => (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setter((prev) => prev ? {
      ...prev,
      imageSize: { width: img.naturalWidth, height: img.naturalHeight },
      displaySize: { width: img.clientWidth, height: img.clientHeight },
    } : prev);
  };

  const handleCapture = async () => {
    setCapturing(true);
    setCaptureError(null);
    try {
      const response = await fetch(`${JETSON_URL}/shot`);
      const data = await response.json().catch(() => null);
      if (!response.ok || typeof data?.front !== 'string' || typeof data?.side !== 'string') {
        setCaptureError(typeof data?.detail === 'string' ? data.detail : 'Jetson 촬영에 실패했습니다.');
        return;
      }
      const frontFile = b64ToFile(data.front, 'front.jpg');
      const sideFile = b64ToFile(data.side, 'side.jpg');
      const previewOf = (file: File): ShotView => ({
        preview: URL.createObjectURL(file),
        results: null,
        error: null,
        imageSize: emptyBox,
        displaySize: emptyBox,
      });
      setFrontShot(previewOf(frontFile));
      setSideShot(previewOf(sideFile));
      const [frontMeasured, sideMeasured] = await Promise.all([
        postMeasure(frontFile, metric),
        postMeasure(sideFile, metric),
      ]);
      setFrontShot((prev) => prev ? { ...prev, results: frontMeasured.results, error: frontMeasured.error } : prev);
      setSideShot((prev) => prev ? { ...prev, results: sideMeasured.results, error: sideMeasured.error } : prev);
    } catch {
      setCaptureError('Jetson에 연결하지 못했습니다.');
    } finally {
      setCapturing(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '50px' }}>
      <h2>손톱 측정</h2>

      <div style={{ display: 'flex', gap: '16px', marginBottom: '12px', fontSize: '14px' }}>
        <label>
          <input
            type="radio"
            name="metric"
            value="chamfer"
            checked={metric === 'chamfer'}
            onChange={() => setMetric('chamfer')}
          />
          {' '}Chamfer
        </label>
        <label>
          <input
            type="radio"
            name="metric"
            value="xor"
            checked={metric === 'xor'}
            onChange={() => setMetric('xor')}
          />
          {' '}XOR
        </label>
      </div>

      <button
        onClick={handleCapture}
        disabled={capturing}
        style={{ padding: '10px 30px', marginBottom: '24px', cursor: capturing ? 'not-allowed' : 'pointer' }}
      >
        {capturing ? '촬영 중' : '촬영'}
      </button>
      {captureError && <p style={{ marginTop: 0, color: '#c00' }}>{captureError}</p>}

      {(frontShot || sideShot) && (
        <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap', justifyContent: 'center', marginBottom: '32px' }}>
          {([
            ['정면', frontShot, bindShotSize(setFrontShot)],
            ['측면', sideShot, bindShotSize(setSideShot)],
          ] as const).map(([label, shot, onLoad]) => shot && (
            <div key={label} style={{ width: '320px' }}>
              <h3 style={{ margin: '0 0 8px' }}>{label}</h3>
              <div style={{ position: 'relative', display: 'block', width: 'fit-content', lineHeight: 0 }}>
                <img
                  src={shot.preview}
                  alt={label}
                  onLoad={onLoad}
                  style={{ maxWidth: '320px', width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }}
                />
                <ContourOverlay results={shot.results} imageSize={shot.imageSize} displaySize={shot.displaySize} />
              </div>
              {shot.error && <p style={{ color: '#c00' }}>{shot.error}</p>}
              <ResultList results={shot.results} />
            </div>
          ))}
        </div>
      )}

      <div style={{ position: 'relative', display: 'inline-block', marginBottom: '20px' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} style={{ marginBottom: '10px', display: 'block' }} />

        {imagePreview && (
          <div style={{ position: 'relative', display: 'block', width: 'fit-content', lineHeight: 0 }}>
            <img
              ref={imageRef}
              src={imagePreview}
              alt="preview"
              onLoad={syncImageSize}
              style={{ maxWidth: '400px', width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }}
            />
            <ContourOverlay results={measurementResults} imageSize={imageSize} displaySize={displaySize} />
          </div>
        )}
      </div>

      <button onClick={handleSubmit} disabled={!imageFile} style={{ padding: '10px 30px', cursor: imageFile ? 'pointer' : 'not-allowed' }}>
        API 전송 및 탐지
      </button>

      {errorMessage && (
        <p style={{ marginTop: '16px', color: '#c00' }}>{errorMessage}</p>
      )}

      {Array.isArray(measurementResults) && (
        <div style={{ marginTop: '30px', textAlign: 'center', width: '100%', maxWidth: '720px' }}>
          <h3>측정 결과</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', justifyContent: 'center' }}>
            {measurementResults.map((res, index) => (
              <div key={index} style={{ margin: '10px 0', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px', width: '160px' }}>
                 {res.preview && (
                   <img
                     src={res.preview}
                     alt={`전처리 ${index + 1}`}
                     style={{ width: '140px', height: '140px', objectFit: 'contain', background: '#000', display: 'block', margin: '0 auto 8px' }}
                   />
                 )}
                 길이 {res.length_mm}mm / 폭 {res.width_mm ? `${res.width_mm}mm` : '측정 불가'}
                 {res.shape ? ` / 쉐입 ${res.shape}` : ''}
                 {res.metric ? ` / ${res.metric === 'xor' ? 'XOR' : 'Chamfer'}` : ''}
                 {res.shape_score != null ? ` (${res.shape_score})` : ''}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
