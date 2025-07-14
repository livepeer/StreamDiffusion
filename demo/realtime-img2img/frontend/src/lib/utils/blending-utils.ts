/**
 * Shared utilities for blending controls
 */

export interface BlendingItem<T> {
  value: T;
  weight: number;
}

/**
 * Normalize weights in a blending list to sum to 1
 * @param items Array of [value, weight] pairs
 * @returns Normalized array of [value, weight] pairs
 */
export function normalizeWeights<T>(items: Array<[T, number]>): Array<[T, number]> {
  const total = items.reduce((sum, [, weight]) => sum + weight, 0);
  if (total <= 0) return items;
  
  return items.map(([value, weight]) => [value, weight / total]);
}

/**
 * Calculate total weight from blending items
 * @param items Array of [value, weight] pairs
 * @returns Total weight sum
 */
export function calculateTotalWeight<T>(items: Array<[T, number]>): number {
  return items.reduce((sum, [, weight]) => sum + weight, 0);
}

/**
 * Update normalize weights setting via API
 * @param endpoint API endpoint for updating normalization setting
 * @param normalize Whether to normalize weights
 * @returns Promise<boolean> Success status
 */
export async function updateNormalizeWeights(endpoint: string, normalize: boolean): Promise<boolean> {
  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ normalize })
    });

    if (!response.ok) {
      const result = await response.json();
      console.error('updateNormalizeWeights: Failed to update normalize weights:', result.detail);
      return false;
    }
    return true;
  } catch (error) {
    console.error('updateNormalizeWeights: Update failed:', error);
    return false;
  }
}

/**
 * Update blending configuration via API
 * @param endpoint API endpoint for updating blending config
 * @param listKey Key name for the list in the request (e.g., 'prompt_list', 'seed_list')
 * @param items Array of [value, weight] pairs
 * @param interpolationMethod Optional interpolation method
 * @returns Promise<boolean> Success status
 */
export async function updateBlendingConfig<T>(
  endpoint: string,
  listKey: string,
  items: Array<[T, number]>,
  interpolationMethod?: string
): Promise<boolean> {
  try {
    const body: any = { [listKey]: items };
    if (interpolationMethod) {
      const methodKey = listKey.replace('_list', '_interpolation_method');
      body[methodKey] = interpolationMethod;
    }

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const result = await response.json();
      console.error('updateBlendingConfig: Failed to update blending config:', result.detail);
      return false;
    }
    return true;
  } catch (error) {
    console.error('updateBlendingConfig: Update failed:', error);
    return false;
  }
} 