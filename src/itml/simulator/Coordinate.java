package itml.simulator;

/**
 *
 *  This class implements the type Coordinate, representing two-dimensional indicies (x,y)
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class Coordinate {

    private int m_x;
    private int m_y;

    /**
     *
     * Constructor, create a new deck.
     *
     * @param  x    An integer representing the x component of the coordinate.
     * @param  y    An integer representing the y component of the coordinate.
     *
     */
    public Coordinate( int x, int y ) {
        m_x = x;
        m_y = y;
    }


    /**
     *
     * Get x component of coordinate.
     *
     * @return  x as an integer
     *
     */
   public int getX() {
        return m_x;
    }

    /**
     *
     * Get y component of coordinate.
     *
     * @return  y as an integer
     *
     */
    public int getY() {
        return m_y;
    }

}

