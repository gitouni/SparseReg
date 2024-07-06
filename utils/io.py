import re
from typing import List, Dict, Tuple

def match_dirs(markered_files:List[str], dir_pattern=r'markered/\w+', sbudir_pattern=r'\d+') -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, Dict[str, List]]]]:
    """extract matched dirs from normal filenames

    Args:
        markered_files (List[str]): List that contains "markered/obj_name/num/normal"

    Raises:
        UserWarning: does not fit 'markered/obj_name' pattern
        UserWarning: does not fit 'markered/obj_name/num' pattern

    Returns:
        Tuple[Dict[str, List[int]], Dict[str, Dict[str, Dict[str, List]]]]: _description_
    """
    matched_dirs = set()
    matched_idxdict = dict()
    matched_subdirs = dict()
    for i, marker_file in enumerate(markered_files):
        name = re.search(dir_pattern, marker_file)  # name of indenter
        if name:
            s_name = name.group()
            matched_dirs.add(s_name)
            if s_name not in matched_idxdict.keys():
                matched_idxdict[s_name] = []
                matched_subdirs[s_name] = dict()
            matched_idxdict[s_name].append(i)
        else:
            raise UserWarning('{} does not fit the dir pattern, causing potential error in subsequent operations.'.format(marker_file))
    print("find matched dirs: {}".format(matched_dirs))
    for s_name in matched_dirs:
        for index in matched_idxdict[s_name]:
            subdir_pattern:str = s_name + r'/' + sbudir_pattern
            name = re.search(subdir_pattern, markered_files[index])  # name + number of indenter
            if name:
                ss_name = name.group()
                if ss_name not in matched_subdirs[s_name]:
                    matched_subdirs[s_name][ss_name] = dict(normal=[],shear=[])
                normal_pattern = ss_name + '/normal'  # name + number of indenter + motion_type
                shear_pattern = ss_name + '/shear'
                if re.search(normal_pattern, markered_files[index]):
                    matched_subdirs[s_name][ss_name]['normal'].append(index)
                elif re.search(shear_pattern, markered_files[index]):
                    matched_subdirs[s_name][ss_name]['shear'].append(index)
                else:
                    raise UserWarning('{} does not fit the dir pattern, causing potential error in subsequent operations.'.format(markered_files[index]))
    return matched_idxdict, matched_subdirs            
    