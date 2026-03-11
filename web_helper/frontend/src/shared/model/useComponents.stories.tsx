import type { Meta, StoryObj } from '@storybook/react';
import { useEffect } from 'react';
import {
  useComponents,
  useComponentsByCategory,
  useComponentSearch,
  useComponentDetails,
  useValidateComponentConfig,
  useComponentStats,
} from './useComponents';
import { apiClient } from '@/shared/api/api-client';
import { mockCategoriesData, mockDetails } from '@/shared/lib/mocks/components';

const setApiMocks = (overrides?: Partial<typeof apiClient>) => {
  apiClient.getAllComponents = async () => mockCategoriesData;
  apiClient.getComponentsByCategory = async () => [mockCategoriesData[0].components[0]];
  apiClient.searchComponents = async () => [mockCategoriesData[0].components[0]];
  apiClient.getComponentDetails = async () => mockDetails;
  apiClient.validateComponentConfig = async () => ({ valid: true });
  if (overrides) {
    Object.assign(apiClient, overrides);
  }
};

const StateBlock = ({ title, state }: { title: string; state: unknown }) => (
  <div className="space-y-2">
    <div className="text-sm font-semibold">{title}</div>
    <pre className="text-xs bg-muted p-2 rounded">{JSON.stringify(state, null, 2)}</pre>
  </div>
);

const UseComponentsStory = () => {
  const result = useComponents();
  return <StateBlock title="useComponents" state={result} />;
};

const UseComponentsByCategoryStory = ({ category }: { category: string }) => {
  const result = useComponentsByCategory(category);
  return <StateBlock title="useComponentsByCategory" state={result} />;
};

const UseComponentSearchStory = ({ query }: { query: string }) => {
  const result = useComponentSearch(query);
  return <StateBlock title="useComponentSearch" state={result} />;
};

const UseComponentDetailsStory = ({ category, name }: { category: string; name: string }) => {
  const result = useComponentDetails(category, name);
  return <StateBlock title="useComponentDetails" state={result} />;
};

const UseValidateComponentConfigStory = ({
  config,
}: {
  config: { type: string; params: Record<string, unknown> };
}) => {
  const mutation = useValidateComponentConfig();
  useEffect(() => {
    mutation.mutate(config);
  }, [config, mutation]);
  return <StateBlock title="useValidateComponentConfig" state={mutation} />;
};

const UseComponentStatsStory = () => {
  const result = useComponentStats();
  return <StateBlock title="useComponentStats" state={result} />;
};

const meta: Meta = {
  title: 'Hooks/useComponents',
};

export default meta;

type Story = StoryObj;

export const ComponentsFetchSuccess: Story = {
  render: () => {
    setApiMocks();
    return <UseComponentsStory />;
  },
};

export const ComponentsFetchError: Story = {
  render: () => {
    setApiMocks({
      getAllComponents: async () => {
        throw new Error('API Error');
      },
    });
    return <UseComponentsStory />;
  },
};

export const ComponentsByCategorySuccess: Story = {
  render: () => {
    setApiMocks();
    return <UseComponentsByCategoryStory category="model" />;
  },
};

export const ComponentsByCategoryEmpty: Story = {
  render: () => {
    setApiMocks();
    return <UseComponentsByCategoryStory category="" />;
  },
};

export const ComponentSearchSuccess: Story = {
  render: () => {
    setApiMocks();
    return <UseComponentSearchStory query="resnet" />;
  },
};

export const ComponentSearchEmpty: Story = {
  render: () => {
    setApiMocks();
    return <UseComponentSearchStory query="" />;
  },
};

export const ComponentDetailsSuccess: Story = {
  render: () => {
    setApiMocks();
    return <UseComponentDetailsStory category="model" name="resnet18" />;
  },
};

export const ComponentDetailsEmpty: Story = {
  render: () => {
    setApiMocks();
    return <UseComponentDetailsStory category="" name="" />;
  },
};

export const ValidateConfigSuccess: Story = {
  render: () => {
    setApiMocks();
    return (
      <UseValidateComponentConfigStory
        config={{ type: 'model.resnet18', params: { num_classes: 10 } }}
      />
    );
  },
};

export const ValidateConfigError: Story = {
  render: () => {
    setApiMocks({
      validateComponentConfig: async () => {
        throw new Error('Validation failed');
      },
    });
    return (
      <UseValidateComponentConfigStory
        config={{ type: 'invalid.component', params: {} }}
      />
    );
  },
};

export const ComponentStatsLoaded: Story = {
  render: () => {
    setApiMocks();
    return <UseComponentStatsStory />;
  },
};

export const ComponentStatsLoading: Story = {
  render: () => {
    apiClient.getAllComponents = async () => new Promise(() => {});
    return <UseComponentStatsStory />;
  },
};
