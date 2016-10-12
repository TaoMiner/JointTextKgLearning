package Common;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
/**
 * Created by ethan on 16/3/23.
 */
public class Constant {
    public static String entity_dic_path = "./etc/mention_list";  //xlore.instance.mention.ttl
    public static String entity_dic_sorted = entity_dic_path+"_cl";
    public static String entity_file = entity_dic_path+"_input";
    public static String entity_trie = Constant.entity_dic_path + "_trie";
    public static String news_path = "./etc/news";
    public static int entity_min_length = 2;
    
    public static Map<String, String> entities = null;
}
