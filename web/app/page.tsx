'use client';

import { useEffect, useState } from 'react';
import styles from './page.module.css';
import Header from '@/components/Header';
import FileUpload from '@/components/FileUpload';
import DatasetCard from '@/components/DatasetCard';
import { Dataset, fetchDatasets, deleteDataset, UploadResponse } from '@/lib/api';

export default function Home() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDatasets = async () => {
    try {
      const data = await fetchDatasets();
      setDatasets(data);
      setError(null);
    } catch (err) {
      setError('Failed to load datasets. Is the API server running?');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  const handleUploadSuccess = (result: UploadResponse) => {
    loadDatasets();
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Delete this dataset and all its data?')) return;
    
    try {
      await deleteDataset(id);
      setDatasets((prev) => prev.filter((d) => d.id !== id));
    } catch (err) {
      alert('Failed to delete dataset');
    }
  };

  return (
    <>
      <Header />
      <main className={styles.main}>
        <div className={styles.container}>
          <section className={styles.hero}>
            <h1 className={styles.title}>
              <span className={styles.titleGradient}>Evaluate</span> Your LLM Conversations
            </h1>
            <p className={styles.subtitle}>
              Upload conversation datasets and detect bad responses through re-ask pattern analysis.
              Powered by CCM, RDM, and LLM Judge detection methods.
            </p>
          </section>

          <section className={styles.uploadSection}>
            <FileUpload onUploadSuccess={handleUploadSuccess} />
          </section>

          <section className={styles.datasetsSection}>
            <div className={styles.sectionHeader}>
              <h2 className={styles.sectionTitle}>Your Datasets</h2>
              <span className={styles.sectionCount}>{datasets.length} total</span>
            </div>

            {loading ? (
              <div className={styles.grid}>
                {[1, 2, 3].map((i) => (
                  <div key={i} className={styles.skeletonCard}>
                    <div className={styles.skeletonHeader}>
                      <div className={`${styles.skeletonIcon} skeleton`} />
                      <div className={styles.skeletonTitleArea}>
                        <div className={`${styles.skeletonTitle} skeleton`} />
                        <div className={`${styles.skeletonDate} skeleton`} />
                      </div>
                    </div>
                    <div className={styles.skeletonStats}>
                      <div className={`${styles.skeletonStat} skeleton`} />
                      <div className={`${styles.skeletonStat} skeleton`} />
                      <div className={`${styles.skeletonStat} skeleton`} />
                    </div>
                  </div>
                ))}
              </div>
            ) : error ? (
              <div className={styles.error}>
                <span className={styles.errorIcon}>⚠️</span>
                <span>{error}</span>
                <button className="btn btn-secondary" onClick={loadDatasets}>
                  Retry
                </button>
              </div>
            ) : datasets.length === 0 ? (
              <div className={styles.empty}>
                <span className={styles.emptyIcon}>📭</span>
                <h3>No datasets yet</h3>
                <p>Upload a CSV or JSON file to get started</p>
              </div>
            ) : (
              <div className={styles.grid}>
                {datasets.map((dataset, index) => (
                  <div
                    key={dataset.id}
                    className="animate-slide-up"
                    style={{ animationDelay: `${index * 0.05}s` }}
                  >
                    <DatasetCard dataset={dataset} onDelete={handleDelete} />
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      </main>
    </>
  );
}

