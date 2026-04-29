import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Landing from './pages/Landing';
import Scan from './pages/Scan';
import Report from './pages/Report';

export default function App() {
  return (
    <>
      <Header />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/scan" element={<Scan />} />
        <Route path="/report/:id" element={<Report />} />
      </Routes>
    </>
  );
}
