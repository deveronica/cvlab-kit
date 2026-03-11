import type { Preview } from '@storybook/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import '../src/index.css';

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });

const preview: Preview = {
  decorators: [
    (Story) => {
      const queryClient = createQueryClient();
      return (
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <Story />
          </BrowserRouter>
        </QueryClientProvider>
      );
    },
  ],
};

export default preview;
