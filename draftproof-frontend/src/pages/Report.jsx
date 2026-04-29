import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getReport } from '../api/draftproofApi';
import ReportPreview from '../components/ReportPreview';

export default function Report() {
  const { id } = useParams();
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getReport(id)
      .then(({ data }) => setReport(data))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="container"><p>Loading report...</p></div>;
  if (!report) return <div className="container"><p>Report not found.</p></div>;

  return (
    <div className="container report-page">
      <h1>Report: {report.document_name}</h1>
      <ReportPreview issues={report.issues} />
    </div>
  );
}
