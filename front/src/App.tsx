import { BrowserRouter, Routes, Route } from 'react-router-dom';
import NailMeasurement from './components/measure';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<NailMeasurement />} />
      </Routes>
    </BrowserRouter>
  );
}