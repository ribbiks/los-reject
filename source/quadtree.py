from source.geometry import Point, LineSegment, BoundingBox


class QuadTree:
    def __init__(self, boundary, capacity):
        self.boundary = boundary
        self.capacity = capacity
        self.segments = []
        self.divided = False
        self.northwest = None
        self.northeast = None
        self.southwest = None
        self.southeast = None

    def insert(self, segment):
        if not self._segment_intersects_box(segment, self.boundary):
            return False

        if len(self.segments) < self.capacity and not self.divided:
            self.segments.append(segment)
            return True

        if not self.divided:
            self.subdivide()

        return (self.northwest.insert(segment) or
                self.northeast.insert(segment) or
                self.southwest.insert(segment) or
                self.southeast.insert(segment))

    def subdivide(self):
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.width / 2
        h = self.boundary.height / 2

        nw = BoundingBox(x, y, w, h)
        ne = BoundingBox(x + w, y, w, h)
        sw = BoundingBox(x, y + h, w, h)
        se = BoundingBox(x + w, y + h, w, h)

        self.northwest = QuadTree(nw, self.capacity)
        self.northeast = QuadTree(ne, self.capacity)
        self.southwest = QuadTree(sw, self.capacity)
        self.southeast = QuadTree(se, self.capacity)

        self.divided = True

    def query(self, search_box):
        found_segments = []

        if not self.boundary.intersects(search_box):
            return found_segments

        for segment in self.segments:
            if self._segment_intersects_box(segment, search_box):
                found_segments.append(segment)

        if self.divided:
            found_segments.extend(self.northwest.query(search_box))
            found_segments.extend(self.northeast.query(search_box))
            found_segments.extend(self.southwest.query(search_box))
            found_segments.extend(self.southeast.query(search_box))

        return found_segments

    def query_bl_tr(self, bottomleft, topright):
        width = topright[0] - bottomleft[0]
        height = topright[1] - bottomleft[1]
        return self.query(BoundingBox(bottomleft[0], bottomleft[1], width, height))

    def _segment_in_boundary(self, segment):
        return (self.boundary.contains(segment.start) or
                self.boundary.contains(segment.end) or
                self._segment_intersects_box(segment, self.boundary))

    @staticmethod
    def _segment_intersects_box(segment, box):
        def compute_outcode(x, y):
            code = 0
            if x < box.x:
                code |= 1  # left
            elif x > box.x + box.width:
                code |= 2  # right
            if y < box.y:
                code |= 4  # top
            elif y > box.y + box.height:
                code |= 8  # bottom
            return code

        x1, y1 = segment.start.x, segment.start.y
        x2, y2 = segment.end.x, segment.end.y
        outcode1 = compute_outcode(x1, y1)
        outcode2 = compute_outcode(x2, y2)

        while True:
            if not (outcode1 | outcode2):  # both points inside the box
                return True
            elif outcode1 & outcode2:  # both points on the same side outside the box
                return False
            else:
                # select an outside point
                outcode_out = outcode1 if outcode1 else outcode2

                # intersect the line with the clipping edge
                if outcode_out & 1:  # left
                    x = box.x
                    y = y1 + (y2 - y1) * (box.x - x1) / (x2 - x1)
                elif outcode_out & 2:  # right
                    x = box.x + box.width
                    y = y1 + (y2 - y1) * (box.x + box.width - x1) / (x2 - x1)
                elif outcode_out & 4:  # top
                    y = box.y
                    x = x1 + (x2 - x1) * (box.y - y1) / (y2 - y1)
                else:  # bottom
                    y = box.y + box.height
                    x = x1 + (x2 - x1) * (box.y + box.height - y1) / (y2 - y1)

                # update the point to the intersection point and continue
                if outcode_out == outcode1:
                    x1, y1 = x, y
                    outcode1 = compute_outcode(x1, y1)
                else:
                    x2, y2 = x, y
                    outcode2 = compute_outcode(x2, y2)


def get_quadtree(all_segments, boundary=(-32768,-32768,65536,65536), capacity=4):
    boundary_box = BoundingBox(boundary[0], boundary[1], boundary[2], boundary[3])
    qt = QuadTree(boundary_box, capacity)
    for sli,segment in enumerate(all_segments):
        qt.insert(LineSegment(Point(segment[0][0], segment[0][1]), Point(segment[1][0], segment[1][1]), metadata=sli))
    return qt
