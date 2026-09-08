import { useState } from 'react';
import NailMeasurement from './components/measure';
import ShapeCompare from './components/classification';

type Tab = 'measure' | 'compare';

export default function App() {
  const [tab, setTab] = useState<Tab>('measure');

  return (
    <div>
      <nav
        style={{
          display: 'flex',
          justifyContent: 'center',
          gap: 8,
          padding: 12,
          borderBottom: '1px solid var(--border)',
        }}
      >
        <button
          type="button"
          onClick={() => setTab('measure')}
          style={{
            padding: '8px 16px',
            fontWeight: tab === 'measure' ? 700 : 400,
            cursor: 'pointer',
          }}
        >
          손톱 측정
        </button>
        <button
          type="button"
          onClick={() => setTab('compare')}
          style={{
            padding: '8px 16px',
            fontWeight: tab === 'compare' ? 700 : 400,
            cursor: 'pointer',
          }}
        >
          형태 비교
        </button>
      </nav>
      {tab === 'measure' ? <NailMeasurement /> : <ShapeCompare />}
    </div>
  );
}
