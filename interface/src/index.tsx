import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';

import { HashRouter } from 'react-router-dom';
import AppRoutes from './routes/AppRoutes';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <HashRouter>
      <AppRoutes />
    </HashRouter>
  </React.StrictMode>
);
