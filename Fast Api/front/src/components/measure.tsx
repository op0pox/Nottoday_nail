import React, { useState, useRef } from 'react';

export default function NailMeasurement() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [measurementResults, setMeasurementResults] = useState<any[] | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState<{ width: number; height: number }>({ width: 0, height: 0 });
  const imageRef = useRef<HTMLImageElement>(null);

  const syncImageSize = () => {
    const img = imageRef.current;
    if (!img) return;
    setImageSize({
      width: img.naturalWidth,
      height: img.naturalHeight,
    });
    setDisplaySize({
      width: img.clientWidth,
      height: img.clientHeight,
    });
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

  const handleImageLoad = () => {
    syncImageSize();
  };

  const handleSubmit = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append('file', imageFile);

    try {
      const response = await fetch('http://localhost:8000/api/measure', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        const detail = data?.detail;
        setMeasurementResults(null);
        setErrorMessage(
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((item: { msg?: string }) => item?.msg ?? JSON.stringify(item)).join('\n')
              : '측정에 실패했습니다.'
        );
        return;
      }
      if (!Array.isArray(data)) {
        setMeasurementResults(null);
        setErrorMessage('예상하지 못한 응답입니다.');
        return;
      }
      setErrorMessage(null);
      setMeasurementResults(data);
    } catch (error) {
      console.error(error);
      setMeasurementResults(null);
      setErrorMessage('서버에 연결하지 못했습니다.');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '50px' }}>
      <h2>손톱 측정</h2>

      <div style={{ position: 'relative', display: 'inline-block', marginBottom: '20px' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} style={{ marginBottom: '10px', display: 'block' }} />

        {imagePreview && (
          <div style={{ position: 'relative', display: 'block', width: 'fit-content', lineHeight: 0 }}>
            <img
              ref={imageRef}
              src={imagePreview}
              alt="preview"
              onLoad={handleImageLoad}
              style={{ maxWidth: '400px', width: '100%', height: 'auto', display: 'block', borderRadius: '4px' }}
            />

            {Array.isArray(measurementResults) && imageSize.width > 0 && displaySize.width > 0 && (
              <svg
                width={displaySize.width}
                height={displaySize.height}
                viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                preserveAspectRatio="xMidYMid meet"
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  pointerEvents: 'none',
                }}
              >
                {measurementResults.map((res, index) => {
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
                    <g key={index}>
                      <path
                        d={d}
                        fill="rgba(255, 0, 85, 0.3)"
                        stroke="#ff0055"
                        strokeWidth="2"
                      />
                    </g>
                  );
                })}
              </svg>
            )}
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
        <div style={{ marginTop: '30px', textAlign: 'center', width: '350px' }}>
          <h3>측정 결과</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {measurementResults.map((res, index) => (
              <li key={index} style={{ margin: '10px 0', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px' }}>
                 길이 {res.length_mm}mm / 폭 {res.width_mm ? `${res.width_mm}mm` : '측정 불가'}
                 {res.shape ? ` / 쉐입 ${res.shape}` : ''}
                 {res.shape_score != null ? ` (${res.shape_score})` : ''}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}