import React from "react";
import { memo, useState } from 'react';
import { Search, Filter, X, Tag} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from './card';
import { Button } from './button';
import { Badge } from './badge';

interface SearchFilter {
  id: string;
  label: string;
  type: 'text' | 'select' | 'date' | 'number';
  options?: string[];
  value: string;
}

interface AdvancedSearchProps {
  onSearch: (filters: SearchFilter[]) => void;
  placeholder?: string;
}

const AdvancedSearch = memo(function AdvancedSearch({
  onSearch,
  placeholder = "Search experiments..."
}: AdvancedSearchProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilters, setActiveFilters] = useState<SearchFilter[]>([]);

  const availableFilters: Omit<SearchFilter, 'value'>[] = [
    { id: 'status', label: 'Status', type: 'select', options: ['completed', 'running', 'failed', 'pending'] },
    { id: 'project', label: 'Project', type: 'text' },
    { id: 'date_range', label: 'Date Range', type: 'date' },
    { id: 'duration', label: 'Duration (min)', type: 'number' },
    { id: 'model', label: 'Model Type', type: 'text' },
  ];

  const addFilter = (filterDef: Omit<SearchFilter, 'value'>) => {
    const newFilter: SearchFilter = { ...filterDef, value: '' };
    setActiveFilters(prev => [...prev, newFilter]);
  };

  const updateFilter = (id: string, value: string) => {
    setActiveFilters(prev =>
      prev.map(filter => filter.id === id ? { ...filter, value } : filter)
    );
  };

  const removeFilter = (id: string) => {
    setActiveFilters(prev => prev.filter(filter => filter.id !== id));
  };

  const handleSearch = () => {
    const allFilters = searchQuery ?
      [{ id: 'query', label: 'Search', type: 'text' as const, value: searchQuery }, ...activeFilters] :
      activeFilters;
    onSearch(allFilters);
  };

  const clearAllFilters = () => {
    setSearchQuery('');
    setActiveFilters([]);
    onSearch([]);
  };

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Search className="h-4 w-4" />
            Advanced Search
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            <Filter className="h-4 w-4" />
            Filters
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Basic Search */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <input
              type="text"
              placeholder={placeholder}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:border-transparent"
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            />
          </div>
          <Button onClick={handleSearch}>
            Search
          </Button>
          {(searchQuery || activeFilters.length > 0) && (
            <Button variant="outline" onClick={clearAllFilters}>
              Clear
            </Button>
          )}
        </div>

        {/* Active Filters */}
        {activeFilters.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {activeFilters.map(filter => (
              <Badge key={filter.id} variant="secondary" className="flex items-center gap-1">
                {filter.label}: {filter.value || 'Any'}
                <X
                  className="h-3 w-3 cursor-pointer hover:text-destructive"
                  onClick={() => removeFilter(filter.id)}
                />
              </Badge>
            ))}
          </div>
        )}

        {/* Expanded Filters */}
        {isExpanded && (
          <div className="space-y-4 border-t pt-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {availableFilters
                .filter(filterDef => !activeFilters.find(f => f.id === filterDef.id))
                .map(filterDef => (
                  <Button
                    key={filterDef.id}
                    variant="outline"
                    size="sm"
                    onClick={() => addFilter(filterDef)}
                    className="justify-start"
                  >
                    <Tag className="h-3 w-3 mr-2" />
                    Add {filterDef.label}
                  </Button>
                ))}
            </div>

            {/* Active Filter Inputs */}
            {activeFilters.length > 0 && (
              <div className="space-y-3">
                {activeFilters.map(filter => (
                  <div key={filter.id} className="flex items-center gap-3">
                    <label className="w-24 text-sm font-medium text-muted-foreground">
                      {filter.label}:
                    </label>
                    {filter.type === 'select' && filter.options ? (
                      <select
                        value={filter.value}
                        onChange={(e) => updateFilter(filter.id, e.target.value)}
                        className="flex-1 p-2 border border-border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:border-transparent"
                      >
                        <option value="">Any</option>
                        {filter.options.map(option => (
                          <option key={option} value={option}>{option}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={filter.type === 'number' ? 'number' : filter.type === 'date' ? 'date' : 'text'}
                        value={filter.value}
                        onChange={(e) => updateFilter(filter.id, e.target.value)}
                        placeholder={`Enter ${filter.label.toLowerCase()}...`}
                        className="flex-1 p-2 border border-border rounded-lg bg-background focus:ring-2 focus:ring-primary focus:border-transparent"
                      />
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFilter(filter.id)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
});

export { AdvancedSearch };
export type { SearchFilter };