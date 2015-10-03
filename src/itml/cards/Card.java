package itml.cards;

import itml.simulator.Coordinate;

/**
 *
 *  This class provides the type Card, which represents a playing card for determining an action a player can make.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */

public class Card {

    public static Coordinate coO  = new Coordinate(  0,  0 );
    public static Coordinate coN  = new Coordinate(  0, +1 );
    public static Coordinate coNE = new Coordinate( +1, +1 );
    public static Coordinate coE  = new Coordinate( +1,  0 );
    public static Coordinate coSE = new Coordinate( +1, -1 );
    public static Coordinate coS  = new Coordinate(  0, -1 );
    public static Coordinate coSW = new Coordinate( -1, -1 );
    public static Coordinate coW  = new Coordinate( -1,  0 );
    public static Coordinate coNW = new Coordinate( -1, +1 );
    public static Coordinate coE2 = new Coordinate( +2,  0 );
    public static Coordinate coW2 = new Coordinate( -2,  0 );

    public enum CardActionType { ctMove, ctDefend, ctAttack }

    private String m_name;
    private CardActionType m_type;
    private int m_col;
    private int m_row;
    private int m_staminaPoints;
    private int m_hitPoints;
    private int m_defencePoints;
    private Coordinate [] m_range;

     /**
     *
     * Constructor, create a card.
     *
     * @param  name           Name of card.
     * @param  type           Type of card.
     * @param  col            Relative column movement
     * @param  row            Relative row movement
     * @param  staminaPoints  Change in staminaPoints
     * @param  hitPoints      Change to health.
     * @param  defencePoints  Defence against opponent's hitpoints.
     * @param  range          Relative coordinate the card's attack affects.
     *
     */
    protected Card( String name, CardActionType type,
          int col, int row, int staminaPoints, int hitPoints, int defencePoints, Coordinate [] range ) {
        m_name = name;
        m_type = type;
        m_col = col;
        m_row = row;
        m_staminaPoints = staminaPoints;
        m_hitPoints = hitPoints;
        m_defencePoints = defencePoints;
        m_range = range;
    }

    /**
     *
     * Get the name of the card.
     *
     * @return  Name of card.
     *
     */
    public String getName() {
        return m_name;
    }

    /**
     *
     * Get the type of the card.
     *
     * @return  type of card.
     *
     */
    public CardActionType getType() {
        return m_type;
    }

    /**
     *
     * Get the relative column offset of the card.
     *
     * @return  column offset as <code>int</code>
     *
     */
    public int getCol() {
        return m_col;
    }

    /**
     *
     * Get the relative row offset of the card.
     *
     * @return  row offset as <code>int</code>
     *
     */
    public int getRow() {
        return m_row;
    }

    /**
     *
     * Get the stamina points offset of the card.
     *
     * @return  stamina points as <code>int</code>
     *
     */
    public int getStaminaPoints() {
        return m_staminaPoints;
    }

    /**
     *
     * Get the hit points offset of the card.
     *
     * @return  hit points as <code>int</code>
     *
     */
    public int getHitPoints() {
        return m_hitPoints;
    }

    /**
     *
     * Get the defence points offset of the card.
     *
     * @return  defence points as <code>int</code>
     *
     */
    public int getDefencePoints() {
        return m_defencePoints;
    }

    /**
     *
     * Check whether a particular square is within the attack range of the card.
     *
     * @param  colA     column of agent
     * @param  rowA     row of agent
     * @param  colOA    column of opponent agent
     * @param  rowOA    row of opponent agent
     *
     * @return  <code>true</code> if location of opponent agent in attack range of agent location,
     *          otherwise <code>false</code>.
     */
    public boolean inAttackRange( int colA, int rowA, int colOA, int rowOA ) {

        for ( Coordinate co : m_range ) {
            if ( ((colA + co.getX()) == colOA) && (( rowA + co.getY()) == rowOA) ) {
                return true;
            }
        }

        return false;
    }

}
