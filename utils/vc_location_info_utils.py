"""
Lightweight City Data Reader for querying city metadata (roads, points) from JSON.

Removed Blender/USD dependencies; only JSON metadata near a USD path is used.
"""
import math
import os
import json
from typing import List, Dict, Tuple, Optional, Any

class CityDataReader:
    """Read and query city JSON data (roads, interesting points)"""
    
    def __init__(self, json_file_path: str):
        """Initialize the CityDataReader with a USD file"""
        self.json_file_path = json_file_path
        self.city_name = ""
        self.roads_data = []
        self.interesting_points_data = []
        self.summary_data = {}
        
        # Load metadata from JSON files
        self._load_json_metadata()
        
    def _load_json_metadata(self):
        """Load city data from JSON files"""
        print(f"Loading JSON metadata from: {self.json_file_path}")
        
        # Determine city name from file path
        self.city_name = os.path.splitext(os.path.basename(self.json_file_path))[0]
        # if self.city_name.endswith("_complete_scene"):
        #     self.city_name = self.city_name.replace("_complete_scene", "")
        
        # print(f"City: {self.city_name}")
        
        # Load roads data
        roads_file = os.path.join(self.json_file_path, f"{self.city_name}_roads.json")
        if os.path.exists(roads_file):
            # print(f"🛣️ Loading roads from: {roads_file}")
            with open(roads_file, 'r', encoding='utf-8') as f:
                self.roads_data = json.load(f)
            # print(f"✅ Loaded {len(self.roads_data)} roads")
        else:
            # print(f"⚠️ Roads file not found: {roads_file}")
            self.roads_data = []
        
        # Load interesting points data
        points_file = os.path.join(self.json_file_path, f"{self.city_name}_interesting_points.json")
        if os.path.exists(points_file):
            # print(f"📍 Loading interesting points from: {points_file}")
            with open(points_file, 'r', encoding='utf-8') as f:
                self.interesting_points_data = json.load(f)
            # print(f"✅ Loaded {len(self.interesting_points_data)} interesting points")
        else:
            # print(f"⚠️ Interesting points file not found: {points_file}")
            self.interesting_points_data = []
        
        # Load summary data
        summary_file = os.path.join(self.json_file_path, f"{self.city_name}_summary.json")
        if os.path.exists(summary_file):
            # print(f"📊 Loading summary from: {summary_file}")
            with open(summary_file, 'r', encoding='utf-8') as f:
                self.summary_data = json.load(f)
            # print(f"✅ Loaded summary data")
        else:
            # Create basic summary from loaded data
            self.summary_data = {
                "city": self.city_name,
                "roads_count": len(self.roads_data),
                "interesting_points_count": len(self.interesting_points_data),
                "roads_by_type": {},
                "points_by_category": {},
                "points_by_type": {}
            }
            
            # Count roads by type
            for road in self.roads_data:
                road_type = road.get('type', 'unknown')
                self.summary_data["roads_by_type"][road_type] = self.summary_data["roads_by_type"].get(road_type, 0) + 1
            
            # Count points by category and type
            for point in self.interesting_points_data:
                category = point.get('category', 'unknown')
                point_type = point.get('type', 'unknown')
                self.summary_data["points_by_category"][category] = self.summary_data["points_by_category"].get(category, 0) + 1
                self.summary_data["points_by_type"][point_type] = self.summary_data["points_by_type"].get(point_type, 0) + 1
            
            # print(f"✅ Created summary data from loaded information")
        
        print(f"Successfully loaded city data")

    
    def get_nearest_road(self, x: float, y: float) -> Optional[Dict[str, Any]]:
        """
        Find the nearest road to the given XY coordinates
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Dictionary containing road information and distance, or None if no roads found
        """
        if not self.roads_data:
            return None
        
        min_distance = float('inf')
        nearest_road = None
        nearest_segment_info = None
        
        for road in self.roads_data:
            points = road.get('points', [])
            if len(points) < 2:
                continue
            
            # Check each line segment in the road
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                # Calculate distance from point to line segment
                distance, closest_point = self._point_to_line_distance((x, y), p1[:2], p2[:2])
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_road = road
                    nearest_segment_info = {
                        'segment_start': p1,
                        'segment_end': p2,
                        'closest_point_on_road': closest_point,
                        'distance': distance
                    }
        
        if nearest_road:
            result = nearest_road.copy()
            result['distance_to_road'] = min_distance
            result['closest_point_on_road'] = nearest_segment_info['closest_point_on_road']
            result['segment_info'] = nearest_segment_info
            return result
        
        return None
    
    def get_points_in_radius(self, x: float, y: float, radius: float) -> List[Dict[str, Any]]:
        """
        Find all interesting points within the given radius
        
        Args:
            x: X coordinate (center)
            y: Y coordinate (center)
            radius: Search radius
            
        Returns:
            List of point dictionaries within the radius, sorted by distance
        """
        if not self.interesting_points_data:
            return []
        
        points_in_radius = []
        
        for point in self.interesting_points_data:
            pos = point.get('position', {})
            px = pos.get('x', 0)
            py = pos.get('y', 0)
            
            dx = px - x
            dy = py - y
            distance = math.sqrt(dx * dx + dy * dy)
            
            if distance <= radius:
                point_with_distance = point.copy()
                point_with_distance['distance_from_center'] = distance
                points_in_radius.append(point_with_distance)
        
        # Sort by distance
        points_in_radius.sort(key=lambda p: p['distance_from_center'])
        
        return points_in_radius
    
    def get_road_by_name(self, road_name: str) -> Optional[Dict[str, Any]]:
        """
        Find road by name (case-insensitive partial match)
        
        Args:
            road_name: Name of the road to search for
            
        Returns:
            Road dictionary if found, None otherwise
        """
        if not self.roads_data:
            return None
        
        road_name_lower = road_name.lower()
        
        for road in self.roads_data:
            road_name_field = road.get('name', '')
            if road_name_field and road_name_lower in road_name_field.lower():
                return road
        
        return None
    
    def get_point_by_name(self, point_name: str) -> Optional[Dict[str, Any]]:
        """
        Find interesting point by name (case-insensitive partial match)
        
        Args:
            point_name: Name of the point to search for
            
        Returns:
            Point dictionary if found, None otherwise
        """
        if not self.interesting_points_data:
            return None
        
        point_name_lower = point_name.lower()
        
        for point in self.interesting_points_data:
            point_name_field = point.get('name', '')
            if point_name_field and point_name_lower in point_name_field.lower():
                return point
        
        return None
    
    def find_road_intersection(self, road1_name: str, road2_name: str) -> Optional[Dict[str, Any]]:
        """
        Find intersection between two roads
        
        Args:
            road1_name: Name of the first road
            road2_name: Name of the second road
            
        Returns:
            Dictionary containing intersection information, or None if no intersection found
        """
        road1 = self.get_road_by_name(road1_name)
        road2 = self.get_road_by_name(road2_name)
        
        if not road1 or not road2:
            return None
        
        # Find intersections between road segments
        intersections = []
        
        points1 = road1.get('points', [])
        points2 = road2.get('points', [])
        
        for i in range(len(points1) - 1):
            for j in range(len(points2) - 1):
                seg1_start = points1[i][:2]  # (x, y)
                seg1_end = points1[i + 1][:2]
                seg2_start = points2[j][:2]
                seg2_end = points2[j + 1][:2]
                
                intersection = self._line_segment_intersection(
                    seg1_start, seg1_end, seg2_start, seg2_end
                )
                
                if intersection:
                    # Calculate height at intersection point (interpolate from road segments)
                    height1 = self._interpolate_height(intersection, seg1_start, seg1_end, 
                                                     points1[i][2], points1[i + 1][2])
                    height2 = self._interpolate_height(intersection, seg2_start, seg2_end, 
                                                     points2[j][2], points2[j + 1][2])
                    avg_height = (height1 + height2) / 2
                    
                    intersections.append({
                        'intersection_point': (intersection[0], intersection[1], avg_height),
                        'road1_segment': (i, i + 1),
                        'road2_segment': (j, j + 1),
                        'road1_name': road1.get('name', ''),
                        'road2_name': road2.get('name', ''),
                        'road1_type': road1.get('type', ''),
                        'road2_type': road2.get('type', '')
                    })
        
        if intersections:
            return {
                'roads': [road1.get('name', ''), road2.get('name', '')],
                'intersections': intersections,
                'intersection_count': len(intersections)
            }
        
        return None
    
    def get_city_summary(self) -> Dict[str, Any]:
        """Get summary information about the city"""
        return self.summary_data
    
    def get_roads_by_type(self, road_type: str) -> List[Dict[str, Any]]:
        """Get all roads of a specific type"""
        return [road for road in self.roads_data if road.get('type') == road_type]
    
    def get_points_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all points of a specific category (highway/amenity)"""
        return [point for point in self.interesting_points_data if point.get('category') == category]
    
    def get_points_by_type(self, point_type: str) -> List[Dict[str, Any]]:
        """Get all points of a specific type (e.g., 'traffic_signals', 'hospital')"""
        return [point for point in self.interesting_points_data if point.get('type') == point_type]
    
    # Helper methods
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                               line_start: Tuple[float, float], 
                               line_end: Tuple[float, float]) -> Tuple[float, Tuple[float, float]]:
        """Calculate distance from point to line segment and return closest point on line"""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        # Vector from line_start to point
        dx_p = px - x1
        dy_p = py - y1
        
        # Calculate the projection parameter
        line_length_sq = dx * dx + dy * dy
        if line_length_sq == 0:
            # Line is a point
            distance = math.sqrt(dx_p * dx_p + dy_p * dy_p)
            return distance, line_start
        
        t = max(0, min(1, (dx_p * dx + dy_p * dy) / line_length_sq))
        
        # Closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance to closest point
        distance = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
        
        return distance, (closest_x, closest_y)
    
    def _line_segment_intersection(self, seg1_start: Tuple[float, float], seg1_end: Tuple[float, float],
                                 seg2_start: Tuple[float, float], seg2_end: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find intersection point between two line segments"""
        x1, y1 = seg1_start
        x2, y2 = seg1_end
        x3, y3 = seg2_start
        x4, y4 = seg2_end
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection is within both line segments
        if 0 <= t <= 1 and 0 <= u <= 1:
            # Calculate intersection point
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_x, intersection_y)
        
        return None
    
    def _interpolate_height(self, point: Tuple[float, float], 
                          seg_start: Tuple[float, float], seg_end: Tuple[float, float],
                          height_start: float, height_end: float) -> float:
        """Interpolate height at a point along a line segment"""
        x, y = point
        x1, y1 = seg_start
        x2, y2 = seg_end
        
        # Calculate parameter t along the segment
        dx = x2 - x1
        dy = y2 - y1
        segment_length_sq = dx * dx + dy * dy
        
        if segment_length_sq == 0:
            return height_start
        
        # Project point onto line segment
        t = ((x - x1) * dx + (y - y1) * dy) / segment_length_sq
        t = max(0, min(1, t))  # Clamp to [0, 1]
        
        # Interpolate height
        return height_start + t * (height_end - height_start)


# Example usage and testing functions

def test_city_data_reader(json_file_path: str):
    """Test function to demonstrate CityDataReader functionality"""
    print("🧪 Testing CityDataReader functionality")
    print("=" * 60)
    
    try:
        # Initialize reader
        reader = CityDataReader(json_file_path)
        
        # Get city summary
        summary = reader.get_city_summary()
        print(f"\n📊 City Summary:")
        print(f"City: {summary['city']}")
        print(f"Roads: {summary['roads_count']}")
        print(f"Interesting Points: {summary['interesting_points_count']}")
        
        # Test nearest road finding
        print(f"\n🛣️ Testing nearest road finding:")
        test_coords = [(0, 0), (100, 100), (-50, 50)]
        for x, y in test_coords:
            nearest_road = reader.get_nearest_road(x, y)
            if nearest_road:
                print(f"  Position ({x}, {y}): Nearest road '{nearest_road['name']}' "
                      f"({nearest_road['type']}) - Distance: {nearest_road['distance_to_road']:.2f}m")
            else:
                print(f"  Position ({x}, {y}): No roads found")
        
        # Test radius search
        print(f"\n📍 Testing radius search:")
        for x, y in test_coords:
            points_in_radius = reader.get_points_in_radius(x, y, 200)
            print(f"  Position ({x}, {y}), radius 200m: {len(points_in_radius)} points found")
            for point in points_in_radius[:3]:  # Show first 3
                print(f"    - {point['type']} ({point['category']}): {point['distance_from_center']:.1f}m")
        
        # Test road search by name
        print(f"\n🔍 Testing road search by name:")
        road_names = ["Main", "Broadway", "First"]
        for road_name in road_names:
            road = reader.get_road_by_name(road_name)
            if road:
                print(f"  Found road '{road['name']}' ({road['type']}) - {len(road['points'])} points")
            else:
                print(f"  Road '{road_name}' not found")
        
        # Test point search by name
        print(f"\n🔍 Testing point search by name:")
        point_names = ["Hospital", "School", "Bank"]
        for point_name in point_names:
            point = reader.get_point_by_name(point_name)
            if point:
                pos = point['position']
                print(f"  Found point '{point['name']}' ({point['type']}) at ({pos['x']:.1f}, {pos['y']:.1f})")
            else:
                print(f"  Point '{point_name}' not found")
        
        # Test road intersection
        print(f"\n🚦 Testing road intersection:")
        road_pairs = [("Main", "First"), ("Broadway", "First")]
        for road1_name, road2_name in road_pairs:
            intersection = reader.find_road_intersection(road1_name, road2_name)
            if intersection:
                print(f"  {road1_name} ∩ {road2_name}: {intersection['intersection_count']} intersections")
                for i, inter in enumerate(intersection['intersections'][:2]):  # Show first 2
                    point = inter['intersection_point']
                    print(f"    Intersection {i+1}: ({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f})")
            else:
                print(f"  {road1_name} ∩ {road2_name}: No intersections found")
        
        print(f"\n✅ CityDataReader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")


def main():
    """Main function for testing"""
    # Configuration
    DATA_DIR = "D:/Desktop/ViCo"
    CITY_NAME = "NY"
    json_file_path = os.path.join(DATA_DIR, "generated", CITY_NAME, f"{CITY_NAME}_complete_scene.usd")
    
    if os.path.exists(json_file_path):
        test_city_data_reader(json_file_path)
    else:
        print(f"❌ USD file not found: {json_file_path}")
        print("Please run blender_export_city_to_usd.py first to create the USD file.")


# Run the test
if __name__ == "__main__":
    main()
