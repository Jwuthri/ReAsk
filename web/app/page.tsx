'use client';

import styles from './page.module.css';
import Header from '@/components/Header';

export default function Home() {
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
        </div>
      </main>
    </>
  );
}
