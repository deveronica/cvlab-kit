import type { Meta, StoryObj } from '@storybook/react';
import userEvent from '@testing-library/user-event';
import { ComponentsPage } from './ComponentsPage';
import { apiClient } from '@/shared/api/api-client';

const mockCategoriesData = [
  {
    category: 'model',
    count: 2,
    components: [
      {
        name: 'resnet18',
        type: 'model',
        path: 'cvlabkit/component/model/resnet18.py',
        description: 'ResNet-18 model for image classification',
        parameters: {
          num_classes: { type: 'int', default: '1000', required: false },
          pretrained: { type: 'bool', default: 'False', required: false },
        },
        examples: [
          {
            name: 'cifar10_resnet18',
            type: 'model.resnet18',
            params: { num_classes: 10, pretrained: false },
          },
        ],
      },
      {
        name: 'vgg16',
        type: 'model',
        path: 'cvlabkit/component/model/vgg16.py',
        description: 'VGG-16 model for image classification',
        parameters: {},
        examples: [],
      },
    ],
  },
  {
    category: 'dataset',
    count: 1,
    components: [
      {
        name: 'cifar10',
        type: 'dataset',
        path: 'cvlabkit/component/dataset/cifar10.py',
        description: 'CIFAR-10 dataset',
        parameters: {},
        examples: [],
      },
    ],
  },
];

const setApiMocks = ({
  categories = mockCategoriesData,
  searchResults = [],
  searchDelay,
}: {
  categories?: any[];
  searchResults?: any[];
  searchDelay?: number;
}) => {
  apiClient.getAllComponents = async () => categories;
  apiClient.searchComponents = async () => {
    if (searchDelay) {
      await new Promise((resolve) => setTimeout(resolve, searchDelay));
    }
    return searchResults;
  };
};

const meta: Meta<typeof ComponentsPage> = {
  title: 'Pages/ComponentsPage',
  component: ComponentsPage,
  render: () => {
    setApiMocks({});
    return <ComponentsPage />;
  },
};

export default meta;

type Story = StoryObj<typeof ComponentsPage>;

export const Default: Story = {};

export const LoadingState: Story = {
  render: () => {
    apiClient.getAllComponents = async () => new Promise(() => {});
    apiClient.searchComponents = async () => new Promise(() => {});
    return <ComponentsPage />;
  },
};

export const FilterByCategory: Story = {
  render: () => {
    setApiMocks({});
    return <ComponentsPage />;
  },
  play: async ({ canvasElement }) => {
    const user = userEvent.setup();
    const modelButton = Array.from(canvasElement.querySelectorAll('div[role="button"]')).find(
      (el) => el.getAttribute('aria-label') === 'model category'
    );
    if (modelButton) {
      await user.click(modelButton as Element);
    }
  },
};

export const SearchComponents: Story = {
  render: () => {
    setApiMocks({ searchResults: [mockCategoriesData[0].components[0]] });
    return <ComponentsPage />;
  },
  play: async ({ canvasElement }) => {
    const user = userEvent.setup();
    const input = canvasElement.querySelector<HTMLInputElement>('input[placeholder="Search components..."]');
    if (input) {
      await user.type(input, 'resnet');
    }
  },
};

export const OpenDetailsModal: Story = {
  render: () => {
    setApiMocks({});
    return <ComponentsPage />;
  },
  play: async ({ canvasElement }) => {
    const user = userEvent.setup();
    const viewButton = Array.from(canvasElement.querySelectorAll('button')).find(
      (el) => el.textContent === 'View Details'
    );
    if (viewButton) {
      await user.click(viewButton as Element);
    }
  },
};
