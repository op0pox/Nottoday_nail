import React, { useState, useRef, useEffect } from 'react';

const JETSON_URL = import.meta.env.VITE_JETSON_URL as string | undefined;
const SEG_ATTEMPTS = 20;
const SEG_RETRY_MS = 200;
const SEG_FAIL = 'Nail detection failed';
const API_URL = 'http://localhost:8000';
// 체커보드 방식은 임시 비활성 (서버 api.py 의 BOARD_ENABLED 와 맞춘다)
const BOARD_ENABLED = false;
// 저장 파일명: 순번_손_손가락_side/front.jpg (utils/JetsonCheck.py 와 같은 규칙)
const HANDS = [
  { value: 'L', label: '왼손' },
  { value: 'R', label: '오른손' },
] as const;
const FINGERS = [
  { value: '01', label: '엄지' },
  { value: '02', label: '검지' },
  { value: '03', label: '중지' },
  { value: '04', label: '약지' },
  { value: '05', label: '소지' },
] as const;
type Hand = (typeof HANDS)[number]['value'];
type Finger = (typeof FINGERS)[number]['value'];
const FINGER_KEYS = HANDS.flatMap((h) => FINGERS.map((f) => `${h.value}_${f.value}`));
const SAVE_SHOT_COUNT = FINGER_KEYS.length;

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

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

// utils/data/dataset_jetson.csv 의 한 사람. saved 는 저장된 손가락 목록 (예: L_01).
type CapturePerson = {
  id: string;
  name: string;
  saved: string[];
};

// Jetson 촬영 한 번(측면+정면 한 쌍). 버튼 한 번에 seg 재시도한 만큼 쌓인다.
type ShotPair = {
  attempt: number;
  side: File;
  front: File;
  sideUrl: string;
  frontUrl: string;
  status: 'pending' | 'ok' | 'fail';
  savedAs: string | null;
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

function downloadB64(b64: string, name: string) {
  const url = URL.createObjectURL(b64ToFile(b64, name));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
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

async function pushUserDisplay(views: { label: string; results: MeasureResult[] | null }[]) {
  if (!JETSON_URL) return;
  const body = {
    views: views.map((view) => ({
      label: view.label,
      items: (view.results ?? []).map((res) => ({
        preview: res.preview ?? null,
        length_mm: res.length_mm ?? null,
        width_mm: res.width_mm ?? null,
        shape: res.shape ?? null,
        contours: res.contours ?? null,
      })),
    })),
  };
  try {
    await fetch(`${JETSON_URL}/display`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch {
    // HDMI 화면이 없어도 운영자 측정은 그대로 둔다.
  }
}

function ResultList({ results }: { results: MeasureResult[] }) {
  return (
    <div className="result-list">
      {results.map((res, index) => (
        <div key={index} className="result-card">
          {res.preview && (
            <img
              src={res.preview}
              alt={`전처리 ${index + 1}`}
              className="result-preview"
            />
          )}
          {`길이 ${res.length_mm ?? '-'}mm / 폭 ${res.width_mm ? `${res.width_mm}mm` : '측정 불가'}`}
          {res.shape ? ` / ${res.shape}형입니다` : ''}
        </div>
      ))}
    </div>
  );
}

export default function NailMeasurement() {
  const [inputMode, setInputMode] = useState<InputMode>('file');
  const [scaleMode, setScaleMode] = useState<ScaleMode>(BOARD_ENABLED ? 'board' : 'hardware');
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [sideFile, setSideFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [fingerGroup, setFingerGroup] = useState<FingerGroup>('other');
  const [measurementResults, setMeasurementResults] = useState<MeasureResult[] | null>(null);
  const [shots, setShots] = useState<ShotView[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });
  const imageRef = useRef<HTMLImageElement>(null);
  // 이번 [Jetson 촬영] 에서 찍힌 사진 쌍들. 관리자가 그중 한 쌍을 골라 저장한다.
  const [history, setHistory] = useState<ShotPair[]>([]);
  const [selectedPair, setSelectedPair] = useState<number | null>(null);
  const [people, setPeople] = useState<CapturePerson[]>([]);
  const [personId, setPersonId] = useState('');
  const [hand, setHand] = useState<Hand>('L');
  const [finger, setFinger] = useState<Finger>('01');

  // 손가락을 고르면 분류 모델도 맞춘다 (엄지 → 엄지 모델, 나머지 → 나머지 모델).
  const chooseFinger = (value: Finger) => {
    setFinger(value);
    setFingerGroup(value === '01' ? 'thumb' : 'other');
  };

  const loadPeople = async () => {
    try {
      const response = await fetch(`${API_URL}/api/capture/people`);
      const data = await response.json();
      if (!response.ok || !Array.isArray(data)) {
        setErrorMessage(errorText(data, '저장 명단을 불러오지 못했습니다.'));
        return;
      }
      setPeople(data);
      setPersonId((prev) => (data.some((p: CapturePerson) => p.id === prev) ? prev : ''));
    } catch {
      setErrorMessage('저장 명단을 불러오지 못했습니다.');
    }
  };

  useEffect(() => {
    void loadPeople();
  }, []);

  const selectedPerson = people.find((p) => p.id === personId) ?? null;
  const pickedPair = selectedPair !== null ? history[selectedPair] ?? null : null;
  const fingerKey = `${hand}_${finger}`;
  const saveName = selectedPerson ? `${selectedPerson.id}_${fingerKey}` : '';
  const alreadySaved = !!selectedPerson && selectedPerson.saved.includes(fingerKey);
  const fingerLabel = (h: string, f: string) =>
    `${HANDS.find((x) => x.value === h)?.label ?? h} ${FINGERS.find((x) => x.value === f)?.label ?? f}`;

  const resetHistory = () => {
    setHistory((prev) => {
      prev.forEach((pair) => {
        URL.revokeObjectURL(pair.sideUrl);
        URL.revokeObjectURL(pair.frontUrl);
      });
      return [];
    });
    setSelectedPair(null);
  };

  const handleSaveShot = async () => {
    if (!pickedPair || selectedPair === null || !selectedPerson || busy) return;
    const pairIndex = selectedPair;
    if (alreadySaved && !window.confirm(`${saveName} 사진이 이미 있습니다. 덮어쓸까요?`)) return;
    setBusy(true);
    setErrorMessage(null);
    try {
      const formData = new FormData();
      formData.append('person', selectedPerson.id);
      formData.append('hand', hand);
      formData.append('finger', finger);
      formData.append('overwrite', alreadySaved ? 'true' : 'false');
      formData.append('side', pickedPair.side);
      formData.append('front', pickedPair.front);
      const response = await fetch(`${API_URL}/api/capture/save`, { method: 'POST', body: formData });
      const data = await response.json().catch(() => null);
      if (!response.ok) {
        // 422(항목 누락) / 404(경로 없음)는 서버가 예전 코드로 돌고 있을 때 난다.
        if (response.status === 422 || response.status === 404) {
          setErrorMessage('사진 저장 실패: 서버가 예전 코드로 돌고 있습니다. docker compose down → up 으로 다시 띄워 주세요.');
        } else {
          setErrorMessage(errorText(data, '사진 저장에 실패했습니다.'));
        }
        return;
      }
      if (!data || !Array.isArray(data.saved) || typeof data.key !== 'string') {
        setErrorMessage('사진 저장 응답이 예상과 다릅니다. docker compose down → up 으로 서버를 다시 띄워 주세요.');
        return;
      }
      const saved: string[] = data.saved;
      setPeople((prev) => prev.map((p) => (p.id === data.id ? { ...p, saved } : p)));
      const savedAs = `${data.id}_${data.key}`;
      setHistory((prev) => prev.map((pair, i) => (i === pairIndex ? { ...pair, savedAs } : pair)));
      setNotice(`#${pickedPair.attempt} 사진을 ${savedAs} (${fingerLabel(hand, finger)}) 로 ${data.overwritten ? '덮어썼습니다' : '저장했습니다'}. (${saved.length}/${SAVE_SHOT_COUNT})`);
      // 다음에 찍을 손가락으로 넘어간다 (아직 저장 안 된 첫 손가락).
      const next = FINGER_KEYS.find((key) => !saved.includes(key));
      if (next) {
        const [nextHand, nextFinger] = next.split('_') as [Hand, Finger];
        setHand(nextHand);
        chooseFinger(nextFinger);
      }
    } catch {
      setErrorMessage('서버에 연결하지 못했습니다.');
    } finally {
      setBusy(false);
    }
  };

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
    if (!imageFile || busy) return;
    setBusy(true);
    setErrorMessage(null);
    const measured = await postMeasure(imageFile, scaleMode, fingerGroup, sideFile);
    setShots([]);
    setMeasurementResults(measured.payload?.front ?? null);
    setErrorMessage(measured.error);
    setBusy(false);
    if (measured.payload) {
      const views = [{ label: '측면', results: measured.payload.front }];
      if (measured.payload.side && sideFile) views.push({ label: '정면', results: measured.payload.side });
      void pushUserDisplay(views);
    }
    if (measured.payload?.side && sideFile) {
      const emptySize = { width: 0, height: 0 };
      setShots([
        { label: '측면', preview: URL.createObjectURL(imageFile), results: measured.payload.front, error: null, imageSize: emptySize, displaySize: emptySize },
        { label: '정면', preview: URL.createObjectURL(sideFile), results: measured.payload.side, error: null, imageSize: emptySize, displaySize: emptySize },
      ]);
    }
  };

  const handleJetsonShot = async () => {
    if (!JETSON_URL) {
      setErrorMessage('VITE_JETSON_URL이 없습니다.');
      return;
    }
    setBusy(true);
    setErrorMessage(null);
    setNotice(null);
    setMeasurementResults(null);
    resetHistory();
    try {
      for (let attempt = 1; attempt <= SEG_ATTEMPTS; attempt += 1) {
        setNotice(`세그멘테이션 ${attempt}/${SEG_ATTEMPTS}`);
        const response = await fetch(`${JETSON_URL}/shot`);
        const data = await response.json();
        if (!response.ok || !data?.front || !data?.side) {
          setShots([]);
          setErrorMessage(errorText(data, 'Jetson 촬영에 실패했습니다.'));
          return;
        }
        const sideShot = b64ToFile(data.side, 'side.jpg');
        const frontShot = b64ToFile(data.front, 'front.jpg');
        const emptySize = { width: 0, height: 0 };
        const pairIndex = attempt - 1;
        setHistory((prev) => [...prev, {
          attempt,
          side: sideShot,
          front: frontShot,
          sideUrl: URL.createObjectURL(sideShot),
          frontUrl: URL.createObjectURL(frontShot),
          status: 'pending',
          savedAs: null,
        }]);
        const markPair = (status: ShotPair['status']) =>
          setHistory((prev) => prev.map((pair, i) => (i === pairIndex ? { ...pair, status } : pair)));
        setShots([
          { label: '측면', preview: URL.createObjectURL(sideShot), results: null, error: null, imageSize: emptySize, displaySize: emptySize },
          { label: '정면', preview: URL.createObjectURL(frontShot), results: null, error: null, imageSize: emptySize, displaySize: emptySize },
        ]);
        const measured = await postMeasure(sideShot, scaleMode, fingerGroup, frontShot);
        if (measured.payload) {
          setNotice(null);
          setShots((prev) => prev.map((shot, index) => ({
            ...shot,
            results: index === 0 ? measured.payload!.front : measured.payload!.side,
            error: null,
          })));
          void pushUserDisplay([
            { label: '측면', results: measured.payload.front },
            { label: '정면', results: measured.payload.side },
          ]);
          markPair('ok');
          setSelectedPair(pairIndex);
          return;
        }
        markPair('fail');
        if (measured.error !== SEG_FAIL || attempt === SEG_ATTEMPTS) {
          setNotice(null);
          setErrorMessage(measured.error);
          setShots((prev) => prev.map((shot) => ({ ...shot, results: null, error: measured.error })));
          return;
        }
        await sleep(SEG_RETRY_MS);
      }
    } catch {
      setShots([]);
      setErrorMessage('Jetson에 연결하지 못했습니다.');
    } finally {
      setBusy(false);
    }
  };

  const handleCalibrate = async () => {
    if (!JETSON_URL) {
      setErrorMessage('VITE_JETSON_URL이 없습니다.');
      return;
    }
    if (busy) return;
    setBusy(true);
    setErrorMessage(null);
    setNotice(null);
    try {
      const response = await fetch(`${JETSON_URL}/shot`);
      const data = await response.json();
      if (!response.ok || !data?.front || !data?.side) {
        setErrorMessage(errorText(data, '캘리브레이션 촬영에 실패했습니다.'));
        return;
      }
      downloadB64(data.side, 'side.jpg');
      downloadB64(data.front, 'front.jpg');
      setNotice('측면·정면 원본을 저장했습니다.');
    } catch {
      setErrorMessage('Jetson에 연결하지 못했습니다.');
    } finally {
      setBusy(false);
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

  const showHistory = inputMode === 'jetson' && history.length > 0;

  return (
    <div className={showHistory ? 'admin-layout with-history' : 'admin-layout'}>
    <div className="admin-page">
      <h2 className="admin-title">손톱 측정</h2>

      <div className="panel option-row">
        <div className="option-group">
          <div className="option-label">입력</div>
          <label className="option">
            <input type="radio" name="inputMode" checked={inputMode === 'file'} onChange={() => { setInputMode('file'); setShots([]); setMeasurementResults(null); setErrorMessage(null); }} />
            {' '}사진 파일
          </label>
          <label className="option">
            <input type="radio" name="inputMode" checked={inputMode === 'jetson'} onChange={() => { setInputMode('jetson'); setShots([]); setMeasurementResults(null); setErrorMessage(null); setFingerGroup(finger === '01' ? 'thumb' : 'other'); }} />
            {' '}Jetson
          </label>
        </div>
        <div className="option-group">
          <div className="option-label">실측</div>
          <label className="option">
            <input type="radio" name="scaleMode" checked={scaleMode === 'board'} disabled={!BOARD_ENABLED} onChange={() => setScaleMode('board')} />
            {' '}체커보드{BOARD_ENABLED ? '' : ' (임시 비활성)'}
          </label>
          <label className="option">
            <input type="radio" name="scaleMode" checked={scaleMode === 'hardware'} onChange={() => setScaleMode('hardware')} />
            {' '}하드웨어
          </label>
        </div>

        <div className="option-group">
        <div className="option-label">분류</div>
        <label className="option">
          <input type="radio" name="fingerGroup" checked={fingerGroup === 'thumb'} onChange={() => setFingerGroup('thumb')} />
          {' '}엄지
        </label>
        <label className="option">
          <input type="radio" name="fingerGroup" checked={fingerGroup === 'other'} onChange={() => setFingerGroup('other')} />
          {' '}나머지
        </label>
        </div>
      </div>

      {inputMode === 'file' && (
        <div className="panel upload-panel">
          <label className="file-field">
            <span className="file-label">측면</span>
            <input type="file" accept="image/*" onChange={handleFileChange} />
          </label>
          <label className="file-field">
            <span className="file-label">정면</span>
            <input type="file" accept="image/*" onChange={handleSideChange} />
          </label>
          {imagePreview && shots.length === 0 && (
            <div className="photo-frame">
              <img
                ref={imageRef}
                src={imagePreview}
                alt="preview"
                onLoad={syncImageSize}
                className="photo"
              />
              {Array.isArray(measurementResults) && imageSize.width > 0 && displaySize.width > 0 && (
                <svg
                  width={displaySize.width}
                  height={displaySize.height}
                  viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
                  preserveAspectRatio="xMidYMid meet"
                  className="photo-overlay"
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
          <button className="btn btn-primary" onClick={handleSubmit} disabled={!imageFile || busy}>
            {busy ? '측정 중' : 'API 전송 및 탐지'}
          </button>
        </div>
      )}

      {inputMode === 'jetson' && (
        <div className="panel button-row">
          <button className="btn btn-primary" onClick={handleJetsonShot} disabled={busy}>
            {busy ? '측정 중' : 'Jetson 촬영'}
          </button>
          <button className="btn" onClick={handleCalibrate} disabled={busy}>
            캘리브레이션
          </button>
        </div>
      )}

      {inputMode === 'jetson' && (
        <div className="panel save-panel">
          <div className="save-row">
            <span className="option-label">저장</span>
            <select className="select" value={personId} onChange={(e) => setPersonId(e.target.value)}>
              <option value="">사람 선택</option>
              {people.map((p) => (
                <option key={p.id} value={p.id}>
                  {`(${p.id}) ${p.name || '이름없음'} - ${p.saved.length}/${SAVE_SHOT_COUNT}`}
                </option>
              ))}
            </select>
            <button className="btn btn-small" onClick={() => void loadPeople()} disabled={busy}>
              명단 새로고침
            </button>
          </div>
          <div className="save-row">
            <span className="option-label">손가락</span>
            <select className="select select-small" value={hand} onChange={(e) => setHand(e.target.value as Hand)}>
              {HANDS.map((h) => (
                <option key={h.value} value={h.value}>{h.label}</option>
              ))}
            </select>
            <select className="select select-small" value={finger} onChange={(e) => chooseFinger(e.target.value as Finger)}>
              {FINGERS.map((f) => (
                <option key={f.value} value={f.value}>
                  {`${f.label}${selectedPerson?.saved.includes(`${hand}_${f.value}`) ? ' (저장됨)' : ''}`}
                </option>
              ))}
            </select>
            <button
              className="btn btn-primary"
              onClick={handleSaveShot}
              disabled={busy || !pickedPair || !!pickedPair.savedAs || !selectedPerson}
            >
              {pickedPair && saveName ? `#${pickedPair.attempt} ${alreadySaved ? '덮어쓰기' : '저장'} (${saveName})` : '사진 저장'}
            </button>
          </div>
          <p className="save-hint">
            {!people.length
              ? '명단이 비어 있습니다. utils/JetsonCheck.py 에서 사람을 추가하세요.'
              : !selectedPerson
                ? '저장할 사람을 고르세요.'
                : !history.length
                  ? '손가락을 고르고 Jetson 촬영을 하면 찍힌 사진이 오른쪽에 쌓입니다. (손가락을 고르면 분류 모델도 엄지/나머지로 맞춰집니다)'
                  : !pickedPair
                    ? '오른쪽 촬영 기록에서 저장할 한 쌍을 고르세요.'
                    : pickedPair.savedAs
                      ? `#${pickedPair.attempt} 는 이미 ${pickedPair.savedAs} 로 저장했습니다. 다른 쌍을 고르세요.`
                      : `#${pickedPair.attempt} 측면·정면 사진을 ${saveName} (${fingerLabel(hand, finger)}) 로 ${alreadySaved ? '덮어씁니다' : '저장합니다'}.`}
          </p>
          {selectedPerson && (
            <div className="finger-status">
              {HANDS.map((h) => (
                <div key={h.value} className="finger-status-row">
                  <span className="finger-status-hand">{h.label}</span>
                  {FINGERS.map((f) => {
                    const key = `${h.value}_${f.value}`;
                    const done = selectedPerson.saved.includes(key);
                    return (
                      <button
                        type="button"
                        key={key}
                        className={`finger-chip${done ? ' done' : ''}${key === fingerKey ? ' current' : ''}`}
                        onClick={() => { setHand(h.value); chooseFinger(f.value); }}
                      >
                        {f.label}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {errorMessage && <p className="message message-error">{errorMessage}</p>}
      {notice && <p className="message">{notice}</p>}

      {inputMode === 'file' && shots.length === 0 && Array.isArray(measurementResults) && (
        <div className="panel result-section">
          <h3 className="section-title">측정 결과</h3>
          <ResultList results={measurementResults} />
        </div>
      )}

      {shots.length > 0 && (
        <div className="shot-grid">
          {shots.map((shot, index) => (
            <div key={shot.label} className="panel shot-card">
              <h3 className="section-title">{shot.label}</h3>
              <div className="photo-frame">
                <img
                  src={shot.preview}
                  alt={shot.label}
                  onLoad={bindShotSize(index)}
                  className="photo"
                />
                {Array.isArray(shot.results) && shot.imageSize.width > 0 && shot.displaySize.width > 0 && (
                  <svg
                    width={shot.displaySize.width}
                    height={shot.displaySize.height}
                    viewBox={`0 0 ${shot.imageSize.width} ${shot.imageSize.height}`}
                    preserveAspectRatio="xMidYMid meet"
                    className="photo-overlay"
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
              {shot.error && <p className="message message-error">{shot.error}</p>}
              {Array.isArray(shot.results) && <ResultList results={shot.results} />}
            </div>
          ))}
        </div>
      )}
    </div>

    {showHistory && (
      <aside className="history-panel">
        <h3 className="section-title">촬영 기록 ({history.length})</h3>
        <p className="save-hint">저장할 한 쌍을 고르세요.</p>
        <div className="history-list">
          {history.map((pair, index) => (
            <button
              type="button"
              key={pair.attempt}
              className={index === selectedPair ? 'history-item selected' : 'history-item'}
              onClick={() => setSelectedPair(index)}
            >
              <div className="history-head">
                <span className="history-no">#{pair.attempt}</span>
                <span className={`history-status status-${pair.status}`}>
                  {pair.status === 'ok' ? 'seg 성공' : pair.status === 'fail' ? '미탐지' : '측정 중'}
                </span>
                {pair.savedAs && <span className="history-saved">{pair.savedAs} 저장됨</span>}
              </div>
              <div className="history-thumbs">
                <figure>
                  <img src={pair.sideUrl} alt={`#${pair.attempt} 측면`} />
                  <figcaption>측면</figcaption>
                </figure>
                <figure>
                  <img src={pair.frontUrl} alt={`#${pair.attempt} 정면`} />
                  <figcaption>정면</figcaption>
                </figure>
              </div>
            </button>
          ))}
        </div>
      </aside>
    )}
    </div>
  );
}
