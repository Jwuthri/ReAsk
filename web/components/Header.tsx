'use client';

import styles from './Header.module.css';
import Link from 'next/link';

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.container}>
        <Link href="/" className={styles.logo}>
          <span className={styles.logoIcon}>⚡</span>
          <span className={styles.logoText}>ReAsk</span>
          <span className={styles.logoTag}>Detector</span>
        </Link>
        
        <nav className={styles.nav}>
          <Link href="/" className={styles.navLink}>Dashboard</Link>
          <a 
            href="https://github.com" 
            target="_blank" 
            rel="noopener noreferrer"
            className={styles.navLink}
          >
            Docs
          </a>
        </nav>
      </div>
    </header>
  );
}

