import { useMemo, useState } from 'react';

const SAMPLE_CHOICES = Array.from({ length: 16 }, (_, i) => 100 + i * 20);

type CompareResult = {
  predicted_shape?: string;
  distance: number;
  n_samples: number;
  kept_left: number;
  kept_right: number;
  ranking?: { name: string; distance: number }[];
};

function stem(name: string) {
  return name.replace(/\.[^.]+$/, '');
}

export default function ShapeCompare() {
  const [leftImages, setLeftImages] = useState<{ name: string; url: string }[]>([]);
  const [rightImage, setRightImage] = useState<string | null>(null);
  const [leftJsons, setLeftJsons] = useState<File[]>([]);
  const [rightJson, setRightJson] = useState<File | null>(null);
  const [nSamples, setNSamples] = useState(160);
  const [result, setResult] = useState<CompareResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const canSubmit = useMemo(
    () => leftJsons.length > 0 && Boolean(rightJson),
    [leftJsons, rightJson],
  );

  const handleCompare = async () => {
    if (!rightJson || leftJsons.length === 0) return;
    const formData = new FormData();
    leftJsons.forEach((file) => formData.append('left_json', file));
    formData.append('right_json', rightJson);
    formData.append('n_samples', String(nSamples));

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/compare', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        const detail = data?.detail;
        setResult(null);
        setErrorMessage(
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
              ? detail.map((item: { msg?: string }) => item?.msg ?? JSON.stringify(item)).join('\n')
              : '비교에 실패했습니다.',
        );
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

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: 32, padding: 16 }}>
      <h2>형태 비교</h2>
      <label style={{ marginBottom: 20 }}>
        점 보간 개수
        <select
          value={nSamples}
          onChange={(e) => setNSamples(Number(e.target.value))}
          style={{ marginLeft: 8 }}
        >
          {SAMPLE_CHOICES.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>

      <div style={{ display: 'flex', gap: 32, width: '100%', maxWidth: 1100, justifyContent: 'center', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 280, maxWidth: 560 }}>
          <h3 style={{ marginTop: 0 }}>왼쪽 (정답 여러 개)</h3>
          <label style={{ display: 'block', marginBottom: 8 }}>
            이미지 (여러 장)
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => {
                const files = Array.from(e.target.files || []);
                setLeftImages(files.map((file) => ({ name: file.name, url: URL.createObjectURL(file) })));
              }}
              style={{ display: 'block', marginTop: 4 }}
            />
          </label>
          <label style={{ display: 'block', marginBottom: 8 }}>
            라벨 JSON (여러 개)
            <input
              type="file"
              accept="application/json,.json"
              multiple
              onChange={(e) => setLeftJsons(Array.from(e.target.files || []))}
              style={{ display: 'block', marginTop: 4 }}
            />
          </label>
          <p style={{ fontSize: 14, margin: '0 0 8px' }}>
            {leftJsons.length}개 JSON
            {leftJsons.length > 0 ? `: ${leftJsons.map((f) => stem(f.name)).join(', ')}` : ''}
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {leftImages.map((img) => (
              <img
                key={img.url}
                src={img.url}
                alt={img.name}
                title={img.name}
                style={{ width: 96, height: 96, objectFit: 'cover', borderRadius: 4 }}
              />
            ))}
          </div>
        </div>

        <div style={{ flex: 1, minWidth: 260, maxWidth: 420 }}>
          <h3 style={{ marginTop: 0 }}>오른쪽 (비교 1개)</h3>
          <label style={{ display: 'block', marginBottom: 8 }}>
            이미지
            <input
              type="file"
              accept="image/*"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) setRightImage(URL.createObjectURL(file));
              }}
              style={{ display: 'block', marginTop: 4 }}
            />
          </label>
          <label style={{ display: 'block', marginBottom: 8 }}>
            라벨 JSON
            <input
              type="file"
              accept="application/json,.json"
              onChange={(e) => {
                const file = e.target.files?.[0];
                setRightJson(file ?? null);
              }}
              style={{ display: 'block', marginTop: 4 }}
            />
          </label>
          {rightJson && <p style={{ fontSize: 14, margin: '0 0 8px' }}>{rightJson.name}</p>}
          {rightImage && (
            <img
              src={rightImage}
              alt="오른쪽"
              style={{ maxWidth: '100%', height: 'auto', borderRadius: 4, display: 'block' }}
            />
          )}
          {result && (
            <div style={{ marginTop: 12, textAlign: 'center' }}>
              <p style={{ fontSize: 20, color: 'var(--text-h)', margin: 0 }}>
                결과: {result.predicted_shape} (오차 {result.distance.toFixed(4)})
              </p>
              <p style={{ fontSize: 14, margin: '8px 0 0' }}>보간 {result.n_samples}점</p>
              {Array.isArray(result.ranking) && (
                <ul style={{ listStyle: 'none', padding: 0, margin: '12px 0 0', fontSize: 13, textAlign: 'left' }}>
                  {result.ranking.map((item) => (
                    <li key={item.name} style={{ margin: '4px 0' }}>
                      {item.name}: {item.distance.toFixed(4)}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </div>
      </div>

      <button
        onClick={handleCompare}
        disabled={!canSubmit || loading}
        style={{ marginTop: 24, padding: '10px 30px', cursor: canSubmit && !loading ? 'pointer' : 'not-allowed' }}
      >
        {loading ? '비교 중...' : '비교'}
      </button>

      {errorMessage && <p style={{ marginTop: 16, color: '#c00' }}>{errorMessage}</p>}
    </div>
  );
}
