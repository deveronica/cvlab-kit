import React from "react";
/**
 * BestRunRecommendationCard - Recommends optimal runs using Pareto optimality.
 */

import { useEffect, useState } from 'react';
import {
  Target,
  RefreshCw,
  Trophy,
  Star,
  Medal,
  Info,
  Settings,
  ChevronDown,
  TrendingUp,
  TrendingDown,
  Sparkles,
  Wand2,
} from 'lucide-react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuCheckboxItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from '@/components/ui/dropdown-menu';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import {
  getRecommendationSummary,
  findBestRuns,
  type RecommendationSummaryResponse,
  type RecommendationStrategy,
  type ParetoRecommendation,
} from '@/lib/api/recommendations';
import { devError } from '@/lib/dev-utils';

interface BestRunRecommendationCardProps {
  project: string;
  availableMetrics: string[];
  onRunSelect?: (runName: string) => void;
}

// Type guard to check if a recommendation is a ParetoRecommendation
function _isParetoRecommendation(rec: any): rec is ParetoRecommendation {
  return 'rank' in rec && 'is_pareto_optimal' in rec;
}

export function BestRunRecommendationCard({
  project,
  availableMetrics,
  onRunSelect,
}: BestRunRecommendationCardProps) {
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<RecommendationSummaryResponse | null>(
    null,
  );

  // Customization state
  const [useCustom, setUseCustom] = useState(false);
  const [selectedObjectives, setSelectedObjectives] = useState<string[]>([]);
  const [minimizeFlags, setMinimizeFlags] = useState<Record<string, boolean>>({});
  const [strategy, setStrategy] = useState<RecommendationStrategy>('pareto');
  const [weights, setWeights] = useState<Record<string, number>>({});

  useEffect(() => {
    loadRecommendations();
  }, [project]);

  const loadRecommendations = async () => {
    if (!project) return;

    setLoading(true);
    try {
      if (useCustom && selectedObjectives.length > 0) {
        // Custom mode - use findBestRuns with user-selected objectives
        const minimize = selectedObjectives.map((obj) => minimizeFlags[obj] ?? false);
        const weightArray =
          strategy === 'weighted'
            ? selectedObjectives.map((obj) => weights[obj] ?? 1.0)
            : undefined;

        const result = await findBestRuns({
          project,
          objectives: selectedObjectives,
          strategy,
          minimize,
          weights: weightArray,
          top_k: 5,
        });

        // Transform RecommendationResponse to summary format for display
        setSummary({
          project,
          total_runs: result.total_runs,
          pareto_optimal_count: result.pareto_optimal_count || 0,
          auto_selected_objectives: selectedObjectives, // Use custom objectives
          top_recommendations: (result.recommendations as ParetoRecommendation[]).map((rec) => ({
            run_name: rec.run_name,
            rank: rec.rank,
            objective_values: rec.objective_values,
            is_pareto_optimal: rec.is_pareto_optimal || false,
            dominates_count: rec.dominates_count || 0,
            dominated_by_count: rec.dominated_by_count || 0,
          })),
        });
      } else {
        // Auto mode - use summary endpoint with auto-selection
        const result = await getRecommendationSummary(project, 5);
        setSummary(result);
      }
    } catch (error) {
      devError('Failed to load recommendations:', error);
      setSummary(null);
    } finally {
      setLoading(false);
    }
  };

  const getRankIcon = (rank: number) => {
    switch (rank) {
      case 1:
        return <Trophy className="h-5 w-5 text-yellow-600 dark:text-yellow-400" />;
      case 2:
        return <Medal className="h-5 w-5 text-gray-400 dark:text-gray-500" />;
      case 3:
        return <Medal className="h-5 w-5 text-orange-600 dark:text-orange-400" />;
      default:
        return <Star className="h-4 w-4 text-blue-600 dark:text-blue-400" />;
    }
  };

  const getRankBadgeColor = (rank: number) => {
    switch (rank) {
      case 1:
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200 border-yellow-300 dark:border-yellow-700';
      case 2:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200 border-gray-300 dark:border-gray-700';
      case 3:
        return 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-200 border-orange-300 dark:border-orange-700';
      default:
        return 'bg-blue-50 text-blue-800 dark:bg-blue-950 dark:text-blue-200 border-blue-200 dark:border-blue-800';
    }
  };

  return (
    <Card variant="compact">
      <CardHeader variant="compact">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <Target className="h-4 w-4 text-green-500 flex-shrink-0" />
            <div>
              <CardTitle size="sm">Best Run Recommendations</CardTitle>
              <CardDescription className="mt-0.5 text-xs">
                Pareto-optimal runs based on multi-objective optimization
              </CardDescription>
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={loadRecommendations}
            disabled={loading}
            className="h-6 px-2"
          >
            <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>

      <CardContent variant="compact" className="space-y-3">
        {/* Mode Selection - Toggle Group */}
        <div className="space-y-2">
          <Label className="text-sm font-medium">Optimization Mode</Label>
          <ToggleGroup
            type="single"
            value={useCustom ? 'custom' : 'auto'}
            onValueChange={(value) => {
              if (value === 'custom') {
                setUseCustom(true);
              } else if (value === 'auto') {
                setUseCustom(false);
                setSelectedObjectives([]);
                setMinimizeFlags({});
                setWeights({});
                loadRecommendations();
              }
            }}
            className="grid grid-cols-2 gap-2 w-full"
          >
            <ToggleGroupItem
              value="auto"
              className="flex items-center gap-2 h-auto py-2.5 px-3 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
            >
              <Wand2 className="h-4 w-4 flex-shrink-0" />
              <div className="flex flex-col items-start text-left">
                <span className="text-sm font-medium">Auto</span>
                <span className="text-xs opacity-80">AI-Selected</span>
              </div>
            </ToggleGroupItem>
            <ToggleGroupItem
              value="custom"
              className="flex items-center gap-2 h-auto py-2.5 px-3 data-[state=on]:bg-primary data-[state=on]:text-primary-foreground"
            >
              <Settings className="h-4 w-4 flex-shrink-0" />
              <div className="flex flex-col items-start text-left">
                <span className="text-sm font-medium">Custom</span>
                <span className="text-xs opacity-80">Manual</span>
              </div>
            </ToggleGroupItem>
          </ToggleGroup>
        </div>

        {/* Custom Mode Configuration */}
        {useCustom && (
          <div className="space-y-3 border rounded-lg p-3 bg-muted/20">
            {/* Metric Selection - Dropdown */}
            <div className="space-y-2">
              <Label className="text-sm font-medium">Select Metrics to Optimize</Label>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="w-full justify-between h-auto min-h-[40px] px-3 py-2">
                    <span className="text-sm">
                      {selectedObjectives.length === 0
                        ? 'Select metrics...'
                        : `${selectedObjectives.length} metric${selectedObjectives.length > 1 ? 's' : ''} selected`}
                    </span>
                    <ChevronDown className="h-4 w-4 ml-2 flex-shrink-0" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-80" align="start">
                  <DropdownMenuLabel>Select Metrics</DropdownMenuLabel>
                  <div className="flex gap-2 px-2 pb-2">
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1"
                      onClick={(e) => {
                        e.stopPropagation();
                        const newObjectives = [...availableMetrics];
                        setSelectedObjectives(newObjectives);
                        const newFlags: Record<string, boolean> = {};
                        const newWeights: Record<string, number> = {};
                        newObjectives.forEach((m) => {
                          newFlags[m] = false;
                          newWeights[m] = 1.0;
                        });
                        setMinimizeFlags(newFlags);
                        setWeights(newWeights);
                      }}
                    >
                      All
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedObjectives([]);
                        setMinimizeFlags({});
                        setWeights({});
                      }}
                    >
                      Clear
                    </Button>
                  </div>
                  <DropdownMenuSeparator />
                  <ScrollArea className="h-64">
                    {availableMetrics.map((metric) => (
                      <DropdownMenuCheckboxItem
                        key={metric}
                        checked={selectedObjectives.includes(metric)}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            setSelectedObjectives([...selectedObjectives, metric]);
                            setMinimizeFlags({ ...minimizeFlags, [metric]: false });
                            setWeights({ ...weights, [metric]: 1.0 });
                          } else {
                            setSelectedObjectives(
                              selectedObjectives.filter((m) => m !== metric),
                            );
                            const newFlags = { ...minimizeFlags };
                            delete newFlags[metric];
                            setMinimizeFlags(newFlags);
                            const newWeights = { ...weights };
                            delete newWeights[metric];
                            setWeights(newWeights);
                          }
                        }}
                        className="font-mono text-sm"
                      >
                        {metric}
                      </DropdownMenuCheckboxItem>
                    ))}
                  </ScrollArea>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {/* Selected Metrics with Direction Toggles */}
            {selectedObjectives.length > 0 && (
              <div className="space-y-2">
                <Label className="text-sm font-medium">Optimization Direction</Label>
                <div className="border rounded-md p-2 space-y-2 max-h-48 overflow-y-auto">
                  {selectedObjectives.map((metric) => (
                    <div
                      key={metric}
                      className="flex items-center justify-between p-2 rounded-md bg-muted/30"
                    >
                      <span className="text-sm font-mono truncate flex-1">{metric}</span>
                      <div className="flex gap-1 border rounded-md p-0.5 ml-2">
                        <Button
                          variant={minimizeFlags[metric] ? 'ghost' : 'default'}
                          size="sm"
                          onClick={() =>
                            setMinimizeFlags({ ...minimizeFlags, [metric]: false })
                          }
                          className="h-6 px-2 text-xs gap-1"
                        >
                          <TrendingUp className="h-3 w-3" />
                          Max
                        </Button>
                        <Button
                          variant={minimizeFlags[metric] ? 'default' : 'ghost'}
                          size="sm"
                          onClick={() =>
                            setMinimizeFlags({ ...minimizeFlags, [metric]: true })
                          }
                          className="h-6 px-2 text-xs gap-1"
                        >
                          <TrendingDown className="h-3 w-3" />
                          Min
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Strategy Selection */}
            {selectedObjectives.length > 0 && (
              <div className="space-y-2">
                <Label className="text-sm font-medium">Optimization Strategy</Label>
                <Select
                  value={strategy}
                  onValueChange={(value) => setStrategy(value as RecommendationStrategy)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pareto">Pareto (Non-dominated)</SelectItem>
                    <SelectItem value="weighted">Weighted Sum</SelectItem>
                    <SelectItem value="rank">Rank-based</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            {/* Weights (only for weighted strategy) */}
            {strategy === 'weighted' && selectedObjectives.length > 0 && (
              <div className="space-y-2">
                <Label className="text-sm font-medium">Metric Weights</Label>
                <div className="space-y-2">
                  {selectedObjectives.map((metric) => (
                    <div key={metric} className="flex items-center gap-2">
                      <Label className="text-xs font-mono min-w-[120px]">{metric}</Label>
                      <input
                        type="range"
                        min="0.1"
                        max="5"
                        step="0.1"
                        value={weights[metric] || 1.0}
                        onChange={(e) =>
                          setWeights({ ...weights, [metric]: parseFloat(e.target.value) })
                        }
                        className="flex-1"
                      />
                      <span className="text-xs text-muted-foreground min-w-[40px] text-right">
                        {(weights[metric] || 1.0).toFixed(1)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Apply Button */}
            {selectedObjectives.length > 0 && (
              <Button
                onClick={loadRecommendations}
                disabled={loading}
                className="w-full gap-2"
                variant="default"
              >
                {loading ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <Sparkles className="h-4 w-4" />
                )}
                Apply Custom Objectives
              </Button>
            )}
          </div>
        )}

        {/* Summary Stats */}
        {summary && !summary.message && (
          <>
            <div className="grid grid-cols-3 gap-2">
              <div className="p-2 border rounded-lg">
                <div className="text-xs text-muted-foreground">Total Runs</div>
                <div className="text-xl font-bold">{summary.total_runs}</div>
              </div>
              <div className="p-2 border rounded-lg border-primary bg-primary/5">
                <div className="text-xs text-muted-foreground">Pareto Optimal</div>
                <div className="text-xl font-bold text-primary">
                  {summary.pareto_optimal_count}
                </div>
              </div>
              <div className="p-2 border rounded-lg">
                <div className="text-xs text-muted-foreground">Optimizing</div>
                <div className="text-xl font-bold">
                  {summary.auto_selected_objectives.length}
                </div>
                <div className="text-[10px] text-muted-foreground">objectives</div>
              </div>
            </div>

            {/* Objectives */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 text-muted-foreground" />
                <h4 className="text-sm font-medium">Optimization Objectives</h4>
                {useCustom && selectedObjectives.length > 0 ? (
                  <Badge
                    variant="default"
                    className="text-xs bg-blue-600 dark:bg-blue-400 text-white dark:text-blue-950"
                  >
                    Custom
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-xs">
                    Auto
                  </Badge>
                )}
              </div>
              <div className="flex flex-wrap gap-2">
                {summary.auto_selected_objectives.map((obj) => (
                  <Badge key={obj} variant="outline" className="font-mono">
                    {obj}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Top Recommendations */}
            {summary.top_recommendations.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium">Top Recommendations</h4>
                <div className="space-y-1.5">
                  {summary.top_recommendations.map((rec, _idx) => (
                    <div
                      key={rec.run_name}
                      className={`p-2 border rounded-lg transition-all cursor-pointer hover:shadow-md ${
                        rec.is_pareto_optimal
                          ? 'border-primary bg-primary/10 dark:bg-primary/20 ring-2 ring-primary/20'
                          : ''
                      }`}
                      onClick={() => onRunSelect?.(rec.run_name)}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex items-start gap-2 flex-1 min-w-0">
                          <div className="flex-shrink-0 pt-0.5">
                            {getRankIcon(rec.rank)}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="text-sm font-mono truncate">
                                {rec.run_name}
                              </span>
                            </div>
                            <div className="flex flex-wrap gap-1">
                              <Badge
                                variant="outline"
                                className={getRankBadgeColor(rec.rank)}
                              >
                                Rank #{rec.rank}
                              </Badge>
                              {rec.is_pareto_optimal && (
                                <Badge
                                  variant="default"
                                  className="bg-primary text-primary-foreground"
                                >
                                  ⭐ Pareto Optimal
                                </Badge>
                              )}
                              {rec.dominates_count > 0 && (
                                <Badge variant="outline" className="text-xs">
                                  Dominates {rec.dominates_count}
                                </Badge>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Objective Values */}
                      <div className="mt-2 pt-2 border-t grid grid-cols-2 gap-2 text-xs">
                        {Object.entries(rec.objective_values)
                          .slice(0, 4)
                          .map(([obj, value]) => (
                            <div key={obj} className="flex justify-between">
                              <span className="text-muted-foreground truncate">
                                {obj}:
                              </span>
                              <span className="font-mono ml-2">
                                {typeof value === 'number'
                                  ? value.toFixed(4)
                                  : value}
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Pareto Info */}
            <div className="p-2 bg-muted/30 rounded-lg">
              <div className="flex items-start gap-2">
                <Info className="h-3 w-3 text-muted-foreground mt-0.5 flex-shrink-0" />
                <div className="text-xs text-muted-foreground">
                  <p className="font-medium mb-0.5">About Pareto Optimality</p>
                  <p className="text-[11px]">
                    A run is Pareto-optimal if no other run performs better across{' '}
                    <strong>all</strong> objectives simultaneously. These represent
                    the best trade-offs between competing goals.
                  </p>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Insufficient Data Message */}
        {summary && summary.message && (
          <div className="text-center py-8 text-muted-foreground">
            <Target className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">{summary.message}</p>
          </div>
        )}

        {/* No Data State */}
        {!loading && !summary && (
          <div className="text-center py-8 text-muted-foreground">
            <Target className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No recommendation data available</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
