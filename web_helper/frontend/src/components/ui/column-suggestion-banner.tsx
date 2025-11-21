import React from "react";
/**
 * Column Suggestion Banner
 *
 * Displays automatic column mapping suggestions inline with quick-accept actions.
 * Shows high-confidence suggestions prominently with one-click acceptance.
 */

import { useState } from 'react';
import { Button } from './button';
import { Badge } from './badge';
import { Card, CardContent } from './card';
import { Check, X, ChevronDown, ChevronUp, Sparkles, Info } from 'lucide-react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './tooltip';
import type { ColumnSuggestion } from '../../lib/api/column-mappings';

interface ColumnSuggestionBannerProps {
  suggestions: ColumnSuggestion[];
  onAccept: (suggestion: ColumnSuggestion) => void;
  onReject: (suggestion: ColumnSuggestion) => void;
  onOpenEditor: () => void;
  isLoading?: boolean;
}

export function ColumnSuggestionBanner({
  suggestions,
  onAccept,
  onReject,
  onOpenEditor,
  isLoading = false,
}: ColumnSuggestionBannerProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  if (suggestions.length === 0) {
    return null;
  }

  // Categorize suggestions by confidence
  const highConfidence = suggestions.filter(s => s.confidence_score >= 0.8);
  const mediumConfidence = suggestions.filter(s => s.confidence_score >= 0.5 && s.confidence_score < 0.8);

  const getConfidenceBadge = (score: number) => {
    if (score >= 0.8) {
      return <Badge variant="default" className="bg-green-500 hover:bg-green-600">High {(score * 100).toFixed(0)}%</Badge>;
    } else if (score >= 0.5) {
      return <Badge variant="secondary">Medium {(score * 100).toFixed(0)}%</Badge>;
    } else {
      return <Badge variant="outline">Low {(score * 100).toFixed(0)}%</Badge>;
    }
  };

  const getAlgorithmIcon = (algorithm: string) => {
    switch (algorithm) {
      case 'semantic':
        return '🧠';
      case 'fuzzy':
        return '🔍';
      case 'value_range':
        return '📊';
      case 'distribution':
        return '📈';
      case 'context':
        return '🔗';
      default:
        return '✨';
    }
  };

  return (
    <Card className="border-blue-200 dark:border-blue-800 bg-blue-50/50 dark:bg-blue-950/20">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-start gap-3 flex-1">
            <Sparkles className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold text-blue-900 dark:text-blue-100">
                        Smart Column Mapping Suggestions
                      </h3>
                      <Badge variant="secondary" className="text-[9px] h-4 px-1 bg-blue-100 dark:bg-blue-900">
                        ✨ AI
                      </Badge>
                    </div>
                    <p className="text-sm text-blue-700 dark:text-blue-300 mt-0.5">
                      {highConfidence.length > 0 && (
                        <span className="font-medium">{highConfidence.length} high-confidence</span>
                      )}
                      {highConfidence.length > 0 && mediumConfidence.length > 0 && ' and '}
                      {mediumConfidence.length > 0 && (
                        <span>{mediumConfidence.length} medium-confidence</span>
                      )}
                      {' '}suggestions found
                    </p>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <button className="text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition-colors duration-200">
                          <Info className="h-3.5 w-3.5" />
                        </button>
                      </TooltipTrigger>
                      <TooltipContent side="bottom" className="max-w-sm">
                        <div className="space-y-1.5">
                          <p className="font-medium text-xs">AI-Powered Column Mapping</p>
                          <p className="text-xs text-muted-foreground">
                            AI analyzes column names and values using multiple algorithms (semantic similarity, fuzzy matching, value range analysis) to suggest standardized names for better organization.
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            <strong>Algorithms:</strong> 🧠 Semantic, 🔍 Fuzzy, 📊 Value Range, 📈 Distribution, 🔗 Context
                          </p>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="text-blue-700 dark:text-blue-300"
                >
                  {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </div>

              {isExpanded && (
                <div className="space-y-2">
                  {suggestions.slice(0, 3).map((suggestion, index) => (
                    <div
                      key={`${suggestion.source_column}-${index}`}
                      className="flex items-center justify-between gap-3 bg-card rounded-lg p-3 border border-blue-100 dark:border-blue-900"
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <span className="text-xl" title={suggestion.algorithm}>
                          {getAlgorithmIcon(suggestion.algorithm)}
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <code className="text-sm font-mono bg-muted px-2 py-0.5 rounded">
                              {suggestion.source_column}
                            </code>
                            <span className="text-muted-foreground">→</span>
                            <code className="text-sm font-mono bg-blue-100 dark:bg-blue-900 px-2 py-0.5 rounded font-semibold">
                              {suggestion.target_column}
                            </code>
                            {getConfidenceBadge(suggestion.confidence_score)}
                          </div>
                          <p className="text-xs text-muted-foreground mt-1 truncate">
                            {suggestion.reason}
                          </p>
                        </div>
                      </div>

                      <div className="flex items-center gap-1 flex-shrink-0">
                        <Button
                          size="sm"
                          variant="default"
                          onClick={() => onAccept(suggestion)}
                          disabled={isLoading}
                          className="bg-green-600 hover:bg-green-700"
                        >
                          <Check className="h-4 w-4 mr-1" />
                          Accept
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => onReject(suggestion)}
                          disabled={isLoading}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}

                  {suggestions.length > 3 && (
                    <div className="text-center pt-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={onOpenEditor}
                        className="text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-800"
                      >
                        View all {suggestions.length} suggestions
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {!isExpanded && (
            <Button
              variant="outline"
              size="sm"
              onClick={onOpenEditor}
              className="flex-shrink-0"
            >
              Review All
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
