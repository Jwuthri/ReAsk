'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import styles from './FileUpload.module.css';
import { uploadDataset, UploadResponse } from '@/lib/api';

interface FileUploadProps {
  onUploadSuccess: (result: UploadResponse) => void;
}

export default function FileUpload({ onUploadSuccess }: FileUploadProps) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    setError(null);

    try {
      const result = await uploadDataset(file);
      onUploadSuccess(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }, [onUploadSuccess]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
    },
    maxFiles: 1,
    disabled: uploading,
  });

  return (
    <div className={styles.wrapper}>
      <div
        {...getRootProps()}
        className={`${styles.dropzone} ${isDragActive ? styles.active : ''} ${uploading ? styles.uploading : ''}`}
      >
        <input {...getInputProps()} />
        
        <div className={styles.content}>
          {uploading ? (
            <>
              <div className={styles.spinner} />
              <span className={styles.title}>Uploading...</span>
            </>
          ) : isDragActive ? (
            <>
              <span className={styles.icon}>📥</span>
              <span className={styles.title}>Drop your file here</span>
            </>
          ) : (
            <>
              <span className={styles.icon}>📁</span>
              <span className={styles.title}>Drop your dataset here</span>
              <span className={styles.subtitle}>or click to browse</span>
              <div className={styles.formats}>
                <span className={styles.format}>.csv</span>
                <span className={styles.format}>.json</span>
              </div>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className={styles.error}>
          <span>⚠️</span>
          {error}
        </div>
      )}

      <div className={styles.schema}>
        <h4>Required Format</h4>
        <div className={styles.schemaContent}>
          <div className={styles.schemaSection}>
            <span className={styles.schemaLabel}>CSV</span>
            <code>conversation_id, message_index, role, content</code>
          </div>
          <div className={styles.schemaSection}>
            <span className={styles.schemaLabel}>JSON</span>
            <code>{`{"conversations": [{"id": "...", "messages": [...]}]}`}</code>
          </div>
        </div>
      </div>
    </div>
  );
}

