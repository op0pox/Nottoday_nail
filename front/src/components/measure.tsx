import React, { useState, useRef } from 'react';

export default function NailMeasurement() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [hand, setHand] = useState<string>('right');
  const [measurementResults, setMeasurementResults] = useState<any[] | null>(null);
  const [imageSize, setImageSize] = useState<{ width: number; height: number }>({ width: 400, height: 300 });
  const imageRef = useRef<HTMLImageElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setMeasurementResults(null);
    }
  };

  const handleImageLoad = () => {
    if (imageRef.current) {
      setImageSize({
        width: imageRef.current.naturalWidth,
        height: imageRef.current.naturalHeight,
      });
    }
  };

  const handleHandToggle = () => {
    setHand(prev => (prev === 'right' ? 'left' : 'right'));
  };

  const handleSubmit = async () => {
    if (!imageFile) return;

    const formData = new FormData();
    formData.append('file', imageFile);
    formData.append('hand', hand);

    try {
      const response = await fetch('http://localhost:8000/api/measure', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setMeasurementResults(data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '50px' }}>
      <h2>손톱 측정</h2>

      <div style={{ display: 'flex', alignItems: 'center', gap: '20px', marginBottom: '20px' }}>
        <span style={{ fontWeight: hand === 'left' ? 'bold' : 'normal' }}>왼손</span>
        <button onClick={handleHandToggle} style={{ padding: '10px 20px', cursor: 'pointer' }}>
          {hand === 'right' ? '오른손 활성화' : '왼손 활성화'}
        </button>
        <span style={{ fontWeight: hand === 'right' ? 'bold' : 'normal' }}>오른손</span>
      </div>

      <div style={{ position: 'relative', display: 'inline-block', marginBottom: '20px' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} style={{ marginBottom: '10px', display: 'block' }} />

        {imagePreview && (
          <div style={{ position: 'relative', display: 'inline-block' }}>
            <img
              ref={imageRef}
              src={imagePreview}
              alt="preview"
              onLoad={handleImageLoad}
              style={{ maxWidth: '400px', display: 'block', borderRadius: '4px' }}
            />

            {measurementResults && (
              <svg
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                }}
                viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                preserveAspectRatio="none"
              >
                {measurementResults.map((res, index) => {
                  if (!res.contours) return null;

                  const d = res.contours.map((cnt: number[][]) =>
    `M ${cnt.map((p: number[]) => p.join(',')).join(' L ')} Z`
  ).join(' ');

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

      {measurementResults && (
        <div style={{ marginTop: '30px', textAlign: 'center', width: '350px' }}>
          <h3>측정 결과</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {measurementResults.map((res, index) => (
              <li key={index} style={{ margin: '10px 0', padding: '10px', backgroundColor: '#f9f9f9', borderRadius: '4px' }}>
                <strong>{res.finger}</strong> : 길이 {res.length_mm}mm / 폭 {res.width_mm ? `${res.width_mm}mm` : '측정 불가'}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}