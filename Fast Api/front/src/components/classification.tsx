import { useState } from 'react';

type ClassifyResult = {
  shape: string;
  template?: string;
  distance?: number;
};

type LooWrong = {
  file: string;
  truth: string;
  predicted: string;
  matched: string;
  distance: number;
};

type LooResult = {
  total: number;
  correct: number;
  wrong: LooWrong[];
  skipped: string[];
};

type VizStage = {
  step: number;
  title: string;
  image: string;
};

type VizResult = {
  green_name: string;
  red_name: string;
  distance?: number;
  stages: VizStage[];
};

function errorDetail(data: { detail?: unknown }, fallback: string) {
  const detail = data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map((item: { msg?: string }) => item?.msg ?? JSON.stringify(item)).join('\n');
  }
  return fallback;
}

export default function ShapeClassify() {
  const [jsonFile, setJsonFile] = useState<File | null>(null);
  const [greenFile, setGreenFile] = useState<File | null>(null);
  const [redFile, setRedFile] = useState<File | null>(null);
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [looResult, setLooResult] = useState<LooResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [looLoading, setLooLoading] = useState(false);
  const [vizLoading, setVizLoading] = useState(false);
  const [vizResult, setVizResult] = useState<VizResult | null>(null);

  const handleClassify = async () => {
    if (!jsonFile) return;
    const formData = new FormData();
    formData.append('file', jsonFile);

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/classify-label', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setResult(null);
        setErrorMessage(errorDetail(data, '분류에 실패했습니다.'));
        return;
      }
      setErrorMessage(null);
      setResult(data);
    } catch {
      setResult(null);
      setErrorMessage('서버에 연결하지 못했습니다.');
    } finally {
      setLoading(false);
    }
  };

  const handleLoo = async () => {
    setLooLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/classify-loo', {
        method: 'POST',
      });
      const data = await response.json();
      if (!response.ok) {
        setLooResult(null);
        setErrorMessage(errorDetail(data, '검증에 실패했습니다.'));
        return;
      }
      setErrorMessage(null);
      setLooResult(data);
    } catch {
      setLooResult(null);
      setErrorMessage('서버에 연결하지 못했습니다.');
    } finally {
      setLooLoading(false);
    }
  };

  const handleViz = async () => {
    if (!greenFile || !redFile) return;
    const formData = new FormData();
    formData.append('green_json', greenFile);
    formData.append('red_json', redFile);
    setVizLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/classify-viz', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setVizResult(null);
        setErrorMessage(errorDetail(data, '시각화에 실패했습니다.'));
        return;
      }
      setErrorMessage(null);
      setVizResult(data);
    } catch {
      setVizResult(null);
      setErrorMessage('서버에 연결하지 못했습니다.');
    } finally {
      setVizLoading(false);
    }
  };

  const handleVizShell = async () => {
    if (!greenFile || !redFile) return;
    const formData = new FormData();
    formData.append('green_json', greenFile);
    formData.append('red_json', redFile);
    setVizLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/classify-viz-shell', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        setVizResult(null);
        setErrorMessage(errorDetail(data, '시각화에 실패했습니다.'));
        return;
      }
      setErrorMessage(null);
      setVizResult(data);
    } catch {
      setVizResult(null);
      setErrorMessage('서버에 연결하지 못했습니다.');
    } finally {
      setVizLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 48, padding: 16 }}>
      <h2>분류</h2>
      <label style={{ display: 'block', marginBottom: 16 }}>
        라벨 JSON
        <input
          type="file"
          accept="application/json,.json"
          onChange={(e) => {
            setJsonFile(e.target.files?.[0] ?? null);
            setResult(null);
            setVizResult(null);
            setErrorMessage(null);
          }}
          style={{ display: 'block', marginTop: 8 }}
        />
      </label>
      {jsonFile && <p style={{ fontSize: 14, margin: '0 0 16px' }}>{jsonFile.name}</p>}

      <div style={{ marginBottom: 16, textAlign: 'center' }}>
        <p style={{ margin: '0 0 12px', fontSize: 14 }}>
          시각화: <span style={{ color: '#0b8', fontWeight: 700 }}>1번째 JSON = 초록</span>
          {' / '}
          <span style={{ color: '#c00', fontWeight: 700 }}>2번째 JSON = 빨강</span>
        </p>
        <label style={{ display: 'inline-block', margin: '0 12px 8px' }}>
          1번째 (초록)
          <input
            type="file"
            accept="application/json,.json"
            onChange={(e) => {
              setGreenFile(e.target.files?.[0] ?? null);
              setVizResult(null);
              setErrorMessage(null);
            }}
            style={{ display: 'block', marginTop: 4 }}
          />
        </label>
        <label style={{ display: 'inline-block', margin: '0 12px 8px' }}>
          2번째 (빨강)
          <input
            type="file"
            accept="application/json,.json"
            onChange={(e) => {
              setRedFile(e.target.files?.[0] ?? null);
              setVizResult(null);
              setErrorMessage(null);
            }}
            style={{ display: 'block', marginTop: 4 }}
          />
        </label>
      </div>

      <div style={{ display: 'flex', gap: 12 }}>
        <button
          onClick={handleClassify}
          disabled={!jsonFile || loading}
          style={{ padding: '10px 30px', cursor: jsonFile && !loading ? 'pointer' : 'not-allowed' }}
        >
          {loading ? '분류 중...' : '분류'}
        </button>
        <button
          onClick={handleLoo}
          disabled={looLoading}
          style={{ padding: '10px 30px', cursor: looLoading ? 'not-allowed' : 'pointer' }}
        >
          {looLoading ? '검증 중...' : '검증'}
        </button>
        <button
          onClick={handleViz}
          disabled={!greenFile || !redFile || vizLoading}
          style={{ padding: '10px 30px', cursor: greenFile && redFile && !vizLoading ? 'pointer' : 'not-allowed' }}
        >
          {vizLoading ? '시각화 중...' : '시각화-큐티클'}
        </button>
        <button
          onClick={handleVizShell}
          disabled={!greenFile || !redFile || vizLoading}
          style={{ padding: '10px 30px', cursor: greenFile && redFile && !vizLoading ? 'pointer' : 'not-allowed' }}
        >
          {vizLoading ? '시각화 중...' : '시각화-전체쉘'}
        </button>
      </div>

      {result && (
        <p style={{ fontSize: 40, fontWeight: 700, marginTop: 40, color: 'var(--text-h)' }}>
          {result.shape}
        </p>
      )}

      {looResult && (
        <div style={{ marginTop: 32, width: '100%', maxWidth: 520, textAlign: 'center' }}>
          <p style={{ fontSize: 24, fontWeight: 700, margin: 0 }}>
            맞음 {looResult.correct} / 전체 {looResult.total}
          </p>
          {looResult.skipped?.length > 0 && (
            <p style={{ fontSize: 13, marginTop: 8 }}>
              건너뜀: {looResult.skipped.join(', ')}
            </p>
          )}
          {looResult.wrong.length === 0 ? (
            <p style={{ marginTop: 16 }}>틀린 항목 없음</p>
          ) : (
            <ul style={{ listStyle: 'none', padding: 0, margin: '16px 0 0', textAlign: 'left', fontSize: 14 }}>
              {looResult.wrong.map((item) => (
                <li key={item.file} style={{ margin: '6px 0' }}>
                  {item.file} {item.truth} → {item.predicted} ({item.matched})
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {vizResult && (
        <div style={{ marginTop: 32, width: '100%', maxWidth: 920 }}>
          <p style={{ textAlign: 'center', margin: '0 0 8px', fontSize: 14 }}>
            초록: 1번째 {vizResult.green_name} / 빨강: 2번째 {vizResult.red_name}
            {vizResult.distance != null ? ` / 평균오차 ${vizResult.distance.toFixed(4)}` : ''}
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, justifyContent: 'center' }}>
            {vizResult.stages.map((stage) => (
              <div key={stage.step} style={{ textAlign: 'center' }}>
                <p style={{ margin: '0 0 8px', fontSize: 14 }}>{stage.title}</p>
                <img
                  src={`data:image/png;base64,${stage.image}`}
                  alt={stage.title}
                  style={{ width: 210, height: 210, background: '#fff', border: '1px solid #ddd' }}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {errorMessage && <p style={{ marginTop: 16, color: '#c00' }}>{errorMessage}</p>}
    </div>
  );
}
