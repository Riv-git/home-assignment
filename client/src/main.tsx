import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Provider as StoreProvider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import App from '@src/App';
import SuspenseUntilReady from './components/SuspenseUntilReady';
import DarkThemeProvider from './providers/DarkThemeProvider';
import { createStore } from './store';
import './index.css';

const store = createStore();

function Client() {
  return (
    <StrictMode>
      <SuspenseUntilReady
        asyncFn={async () => {
          console.log('Application is up and running!');
        }}
      >
        <StoreProvider store={store}>
          <BrowserRouter>
            <DarkThemeProvider>
              <App />
            </DarkThemeProvider>
          </BrowserRouter>
        </StoreProvider>
      </SuspenseUntilReady>
    </StrictMode>
  );
}

const rootElement = document.getElementById('root')!;

createRoot(rootElement).render(<Client />);
